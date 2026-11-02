import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
N = int(os.environ.get('MC_DRAWS', '10000'))
MIN_VALUE = 0.0001
RANDOM_SEED = 42
STAT_NAMES = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
SOURCES = ['Production', 'Manufacture']
FATES = ['soil', 'ocean', 'river']
INDICATORS = [f'{s}_MP' for s in SOURCES] + ['total_MP'] + [f'{s}_{f}' for s in SOURCES for f in FATES] + [f'{s}_environment' for s in SOURCES] + [f'total_{f}' for f in FATES] + ['total_environment']
SCHEMES = {'Scheme_1_Mild_tilt': {'input': BASE_DIR / 'Scheme_1_Mild_tilt/dataset/2019_manufacturing.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_1_Mild_tilt' / 'manufacturing'}, 'Scheme_2_Moderate_tilt': {'input': BASE_DIR / 'Scheme_2_Moderate_tilt/dataset/2019_manufacturing.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_2_Moderate_tilt' / 'manufacturing'}, 'Scheme_3_Strong_tilt': {'input': BASE_DIR / 'Scheme_3_Strong_tilt/dataset/2019_manufacturing.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_3_Strong_tilt' / 'manufacturing'}}

def normal_positive(mean, cv, size=N):
    mean = pd.to_numeric(mean, errors='coerce')
    mean = 0.0 if pd.isna(mean) else float(mean)
    values = np.random.normal(mean, abs(cv * mean), size)
    values[values < 0] = MIN_VALUE
    return values

def normal_fraction(mean, cv, size=N):
    return np.clip(normal_positive(mean, cv, size), 0, 1)

def normalize_arrays(arrays):
    total = np.sum(arrays, axis=0)
    total[total <= 0] = MIN_VALUE
    return [array / total for array in arrays]

def stat(values):
    return [np.mean(values), np.std(values), *np.quantile(values, [0.05, 0.25, 0.75, 0.95])]

def build_result_table(data):
    columns = [f'{indicator}_{stat_name}' for indicator in INDICATORS for stat_name in STAT_NAMES]
    return pd.DataFrame(np.nan, index=range(len(data)), columns=columns)

def add_base_info(result, data):
    result.insert(0, 'city', data['city'])
    result.insert(1, 'Category', data['Category'])
    result.insert(2, 'level', data['level'])
    result.insert(3, 'province', data['province'])
    return result

def write_stat_sheets(result, output_file):
    base = result[['city', 'Category', 'level', 'province']]
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for stat_name in STAT_NAMES:
            columns = [col for col in result.columns if col.endswith('_' + stat_name)]
            sheet = pd.concat([base, result[columns]], axis=1)
            sheet = sheet.rename(columns={col: col.removesuffix('_' + stat_name) for col in columns})
            sheet.to_excel(writer, sheet_name=stat_name, index=False)

def organize_statistics(mc_file, grouped_file, scheme_name):
    with pd.ExcelWriter(grouped_file, engine='openpyxl') as writer:
        for stat_name in STAT_NAMES:
            print(f'Summarizing: {scheme_name}—{stat_name}')
            df = pd.read_excel(mc_file, sheet_name=stat_name)
            missing = [col for col in ['city', 'Category', 'level', *INDICATORS] if col not in df.columns]
            if missing:
                raise KeyError(f'{scheme_name}—{stat_name} missing fields: {missing}')
            df[INDICATORS] = df[INDICATORS].apply(pd.to_numeric, errors='coerce')
            if df[INDICATORS].isna().any().any():
                raise ValueError(f'{scheme_name}—{stat_name} contains invalid numeric values')
            city_order = df[['city']].drop_duplicates().reset_index(drop=True)
            city_order['city_order'] = range(len(city_order))
            level_group = df.groupby('level', as_index=False, dropna=False)[INDICATORS].sum().sort_values('level')
            city_group = df.groupby('city', as_index=False, dropna=False)[INDICATORS].sum()
            city_group = city_group.merge(city_order, on='city', how='left', validate='one_to_one')
            city_group = city_group.sort_values('city_order').drop(columns='city_order')
            city_type_group = df.groupby('Category', as_index=False, dropna=False)[INDICATORS].sum().sort_values('Category')
            city_level_group = df.groupby(['Category', 'level'], as_index=False, dropna=False)[INDICATORS].sum()
            city_level_group = city_level_group.sort_values(['Category', 'level'])
            original_total = df[INDICATORS].sum().to_numpy()
            groups = {'level': level_group, 'city': city_group, 'city_type': city_type_group, 'city_level': city_level_group}
            for name, grouped in groups.items():
                if not np.allclose(grouped[INDICATORS].sum().to_numpy(), original_total, rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f'{scheme_name}—{stat_name}—{name} grouped sum check failed')
                grouped.to_excel(writer, sheet_name=f'{stat_name}_{name}', index=False)
    print(f'Grouped results saved: {grouped_file}')

def run_scheme(scheme_name, input_file, output_dir):
    input_file, output_dir = (Path(input_file), Path(output_dir))
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_file = output_dir / 'prod_manu_city_pop_result_MC.xlsx'
    grouped_file = output_dir / 'prod_manu_all_stats_grouped_result.xlsx'
    print('\n' + '=' * 80)
    print(f'Processing: {scheme_name}')
    print(f'Input file: {input_file}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    data = pd.read_excel(input_file, sheet_name='prod-manu').reset_index(drop=True)
    required = ['city', 'Category', 'level', 'province', 'pro_cbe/kt', 'manu_cbe/kt', 'AG11', 'AG12', 'TC16', 'TC13', 'TC14', 'TC15', 'AG21', 'AG22', 'TC26', 'TC23', 'TC24', 'TC25', 'TC315', 'TC414', 'TC57', 'TC58', 'TC59', 'TC510', 'TC614', 'TC615', 'TC714', 'TC814', 'TC811', 'TC914', 'TC911', 'TC1014', 'TC1011', 'TC1112', 'TC1113', 'TC1215', 'TC1417', 'TC1418', 'TC1516', 'TC1517', 'TC1718', 'TC1719']
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f'{scheme_name}Input file missing fields: {missing}')
    print(f'Input dimensions: {data.shape}')
    np.random.seed(RANDOM_SEED)
    result = build_result_table(data)
    for i, row in data.iterrows():
        initial = {'Production': normal_positive(row['pro_cbe/kt'], 0.16), 'Manufacture': normal_positive(row['manu_cbe/kt'], 0.16)}
        A3 = {s: np.zeros(N) for s in SOURCES}
        A4 = {s: np.zeros(N) for s in SOURCES}
        A5 = {s: np.zeros(N) for s in SOURCES}
        A6 = {s: np.zeros(N) for s in SOURCES}
        A14 = {s: np.zeros(N) for s in SOURCES}
        A15 = {s: np.zeros(N) for s in SOURCES}
        A16 = {s: np.zeros(N) for s in SOURCES}
        A17 = {s: np.zeros(N) for s in SOURCES}
        A18 = {s: np.zeros(N) for s in SOURCES}
        A19 = {s: np.zeros(N) for s in SOURCES}
        RAG11, RAG12, RTC16 = normalize_arrays([normal_positive(row['AG11'], 0.43), normal_positive(row['AG12'], 0.43), normal_positive(row['TC16'], 0.43)])
        G1_prod = initial['Production'] * RAG11
        G2_prod = initial['Production'] * RAG12
        A6['Production'] += initial['Production'] * RTC16
        RTC13, RTC14, RTC15 = normalize_arrays([normal_positive(row['TC13'], 0.26), normal_positive(row['TC14'], 0.25), normal_positive(row['TC15'], 0.43)])
        RTC23, RTC24, RTC25 = normalize_arrays([normal_positive(row['TC23'], 0.26), normal_positive(row['TC24'], 0.25), normal_positive(row['TC25'], 0.43)])
        A3['Production'] += G1_prod * RTC13 + G2_prod * RTC23
        A4['Production'] += G1_prod * RTC14 + G2_prod * RTC24
        A5['Production'] += G1_prod * RTC15 + G2_prod * RTC25
        RAG21, RAG22, RTC26 = normalize_arrays([normal_positive(row['AG21'], 0.43), normal_positive(row['AG22'], 0.43), normal_positive(row['TC26'], 0.43)])
        G1_manu = initial['Manufacture'] * RAG21
        G2_manu = initial['Manufacture'] * RAG22
        A6['Manufacture'] += initial['Manufacture'] * RTC26
        A3['Manufacture'] += G1_manu * RTC13 + G2_manu * RTC23
        A4['Manufacture'] += G1_manu * RTC14 + G2_manu * RTC24
        A5['Manufacture'] += G1_manu * RTC15 + G2_manu * RTC25
        TC315 = normal_fraction(row['TC315'], 0.05)
        TC414 = normal_fraction(row['TC414'], 0.05)
        RTC614, RTC615 = normalize_arrays([normal_positive(row['TC614'], 0.25), normal_positive(row['TC615'], 0.25)])
        for source in SOURCES:
            A15[source] += A3[source] * TC315
            A14[source] += A4[source] * TC414 + A6[source] * RTC614
            A15[source] += A6[source] * RTC615
        RTC57, RTC58, RTC59, RTC510 = normalize_arrays([normal_positive(row['TC57'], 0.16), normal_positive(row['TC58'], 0.16), normal_positive(row['TC59'], 0.16), normal_positive(row['TC510'], 0.16)])
        TC714 = normal_fraction(row['TC714'], 0.05)
        RTC814, RTC811 = normalize_arrays([normal_positive(row['TC814'], 0.16), normal_positive(row['TC811'], 0.16)])
        RTC914, RTC911 = normalize_arrays([normal_positive(row['TC914'], 0.16), normal_positive(row['TC911'], 0.16)])
        RTC1014, RTC1011 = normalize_arrays([normal_positive(row['TC1014'], 0.16), normal_positive(row['TC1011'], 0.16)])
        RTC1112, RTC1113 = normalize_arrays([normal_positive(row['TC1112'], 0.16), normal_positive(row['TC1113'], 0.16)])
        TC1215 = normal_fraction(row['TC1215'], 0.05)
        for source in SOURCES:
            A7, A8 = (A5[source] * RTC57, A5[source] * RTC58)
            A9, A10 = (A5[source] * RTC59, A5[source] * RTC510)
            A14[source] += A7 * TC714
            A14[source] += A8 * RTC814 + A9 * RTC914 + A10 * RTC1014
            A11 = A8 * RTC811 + A9 * RTC911 + A10 * RTC1011
            A12 = A11 * RTC1112
            A15[source] += A12 * TC1215
        RTC1417, RTC1418 = normalize_arrays([normal_positive(row['TC1417'], 0.42), normal_positive(row['TC1418'], 0.42)])
        RTC1516, RTC1517 = normalize_arrays([normal_positive(row['TC1516'], 0.46), normal_positive(row['TC1517'], 0.46)])
        RTC1718, RTC1719 = normalize_arrays([normal_positive(row['TC1718'], 0.16), normal_positive(row['TC1719'], 0.16)])
        for source in SOURCES:
            A17[source] += A14[source] * RTC1417
            A18[source] += A14[source] * RTC1418
            A16[source] += A15[source] * RTC1516
            A17[source] += A15[source] * RTC1517
            A18[source] += A17[source] * RTC1718
            A19[source] += A17[source] * RTC1719
        total_soil = sum((A16[source] for source in SOURCES))
        total_ocean = sum((A18[source] for source in SOURCES))
        total_river = sum((A19[source] for source in SOURCES))
        values = {f'{source}_MP': initial[source] for source in SOURCES}
        values['total_MP'] = initial['Production'] + initial['Manufacture']
        for source in SOURCES:
            values[f'{source}_soil'] = A16[source]
            values[f'{source}_ocean'] = A18[source]
            values[f'{source}_river'] = A19[source]
            values[f'{source}_environment'] = A16[source] + A18[source] + A19[source]
        values['total_soil'] = total_soil
        values['total_ocean'] = total_ocean
        values['total_river'] = total_river
        values['total_environment'] = total_soil + total_ocean + total_river
        row_result = []
        for indicator in INDICATORS:
            row_result.extend(stat(values[indicator]))
        result.iloc[i] = row_result
        if (i + 1) % 100 == 0 or i + 1 == len(data):
            print(f'\rSimulation progress: {i + 1:,}/{len(data):,}', end='', flush=True)
    print()
    result = add_base_info(result, data)
    write_stat_sheets(result, mc_file)
    print(f'Monte Carlo results saved: {mc_file}')
    organize_statistics(mc_file, grouped_file, scheme_name)
    return {'scheme': scheme_name, 'monte_carlo_results': str(mc_file), 'statistical_summary': str(grouped_file)}
