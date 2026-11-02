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
SOURCES = ['Landfill', 'Incineration', 'Recycling', 'Mismanagement']
FATES = ['soil', 'ocean', 'river']
STAT_NAMES = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
INDICATORS = [f'{s}_MP' for s in SOURCES] + [f'{s}_{f}' for s in SOURCES for f in FATES] + [f'{s}_environment' for s in SOURCES] + [f'total_{f}' for f in FATES] + ['total_environment']
SCHEMES = {'Scheme_1_Mild_tilt': {'input': BASE_DIR / 'Scheme_1_Mild_tilt/dataset/2019_waste.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_1_Mild_tilt' / 'waste'}, 'Scheme_2_Moderate_tilt': {'input': BASE_DIR / 'Scheme_2_Moderate_tilt/dataset/2019_waste.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_2_Moderate_tilt' / 'waste'}, 'Scheme_3_Strong_tilt': {'input': BASE_DIR / 'Scheme_3_Strong_tilt/dataset/2019_waste.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_3_Strong_tilt' / 'waste'}}

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

def read_input(input_file):
    sheets = {name: pd.read_excel(input_file, sheet_name=name).reset_index(drop=True) for name in SOURCES}
    keys = ['city', 'Category', 'level', 'province']
    for name, df in sheets.items():
        missing = [col for col in keys if col not in df.columns]
        if missing:
            raise KeyError(f'{input_file} / {name} missing fields: {missing}')
    base = sheets['Landfill'][keys].astype(str)
    for name in SOURCES[1:]:
        if len(sheets[name]) != len(sheets['Landfill']):
            raise ValueError(f'{name} row count does not match Landfill')
        if not sheets[name][keys].astype(str).equals(base):
            raise ValueError(f'{name} city/category/level/province order does not match Landfill')
    return sheets

def build_result(n):
    columns = [f'{indicator}_{stat_name}' for indicator in INDICATORS for stat_name in STAT_NAMES]
    return pd.DataFrame(np.nan, index=range(n), columns=columns)

def write_stat_sheets(result, data, output_file):
    base = data[['city', 'Category', 'level', 'province']].reset_index(drop=True)
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
            groups = {'level': level_group, 'city': city_group, 'city_type': city_type_group, 'city_level': city_level_group}
            original_total = df[INDICATORS].sum().to_numpy()
            for name, grouped in groups.items():
                grouped_total = grouped[INDICATORS].sum().to_numpy()
                if not np.allclose(grouped_total, original_total, rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f'{scheme_name}—{stat_name}—{name} grouped sum check failed')
                grouped.to_excel(writer, sheet_name=f'{stat_name}_{name}', index=False)
    print(f'Grouped results saved: {grouped_file}')

def run_scheme(scheme_name, input_file, output_dir):
    input_file, output_dir = (Path(input_file), Path(output_dir))
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_file = output_dir / 'waste_env_joint_result_MC.xlsx'
    grouped_file = output_dir / 'waste_env_all_stats_grouped_result.xlsx'
    print('\n' + '=' * 80)
    print(f'Processing: {scheme_name}')
    print(f'Input file: {input_file}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    sheets = read_input(input_file)
    landfill = sheets['Landfill']
    inc = sheets['Incineration']
    rec = sheets['Recycling']
    mis = sheets['Mismanagement']
    n = len(landfill)
    result = build_result(n)
    np.random.seed(RANDOM_SEED)
    for i in range(n):
        initial = {'Landfill': normal_positive(landfill.loc[i, 'Landfill/kt'], 0.26) * normal_positive(landfill.loc[i, 'Landfill-EF'], 0.45), 'Incineration': normal_positive(inc.loc[i, 'Incineration/kt'], 0.26) * normal_positive(inc.loc[i, 'Incineration-EF'], 0.45), 'Recycling': normal_positive(rec.loc[i, 'Recycling/kt'], 0.26) * normal_positive(rec.loc[i, 'Recycling-EF'], 0.45), 'Mismanagement': normal_positive(mis.loc[i, 'Mismanagement/kt'], 0.26) * normal_positive(mis.loc[i, 'Mismanagement-EF'], 0.45)}
        landfill_input = {s: np.zeros(N) for s in SOURCES}
        recycle_input = {s: np.zeros(N) for s in SOURCES}
        E5 = {s: np.zeros(N) for s in SOURCES}
        E6 = {s: np.zeros(N) for s in SOURCES}
        E7 = {s: np.zeros(N) for s in SOURCES}
        E8 = {s: np.zeros(N) for s in SOURCES}
        landfill_input['Landfill'] += initial['Landfill']
        recycle_input['Recycling'] += initial['Recycling']
        REG221, REG222 = normalize_arrays([normal_positive(inc.loc[i, 'EG221'], 0.43), normal_positive(inc.loc[i, 'EG222'], 0.43)])
        G21 = initial['Incineration'] * REG221
        G22 = initial['Incineration'] * REG222
        RGE211, RGE213 = normalize_arrays([normal_positive(inc.loc[i, 'GE211'], 0.43), normal_positive(inc.loc[i, 'GE213'], 0.43)])
        landfill_input['Incineration'] += G21 * RGE211
        recycle_input['Incineration'] += G21 * RGE213
        RGG2223, RGG2224, RTC28 = normalize_arrays([normal_positive(inc.loc[i, 'GG2223'], 0.43), normal_positive(inc.loc[i, 'GG2224'], 0.43), normal_positive(inc.loc[i, 'TC28'], 0.43)])
        G23 = G22 * RGG2223
        G24 = G22 * RGG2224
        E8['Incineration'] += G22 * RTC28
        RGE231, RGE233 = normalize_arrays([normal_positive(inc.loc[i, 'GE231'], 0.43), normal_positive(inc.loc[i, 'GE233'], 0.43)])
        landfill_input['Incineration'] += G23 * RGE231
        recycle_input['Incineration'] += G23 * RGE233
        E7['Incineration'] += G24 * normal_fraction(inc.loc[i, 'TC27'], 0.05)
        REG331, REG332, RTC38 = normalize_arrays([normal_positive(rec.loc[i, 'EG331'], 0.43), normal_positive(rec.loc[i, 'EG332'], 0.43), normal_positive(rec.loc[i, 'TC38'], 0.43)])
        GE311 = normal_fraction(rec.loc[i, 'GE311'], 0.05)
        RTC35, RTC36, RTC37 = normalize_arrays([normal_positive(rec.loc[i, 'TC35'], 0.26), normal_positive(rec.loc[i, 'TC36'], 0.25), normal_positive(rec.loc[i, 'TC37'], 0.43)])
        for source in SOURCES:
            G31 = recycle_input[source] * REG331
            G32 = recycle_input[source] * REG332
            E8[source] += recycle_input[source] * RTC38
            landfill_input[source] += G31 * GE311
            E5[source] += G32 * RTC35
            E6[source] += G32 * RTC36
            E7[source] += G32 * RTC37
        EG11 = normal_fraction(landfill.loc[i, 'EG11'], 0.25)
        RTC15, RTC16, RTC17 = normalize_arrays([normal_positive(landfill.loc[i, 'TC15'], 0.26), normal_positive(landfill.loc[i, 'TC16'], 0.25), normal_positive(landfill.loc[i, 'TC17'], 0.43)])
        for source in SOURCES:
            G1 = landfill_input[source] * EG11
            E5[source] += G1 * RTC15
            E6[source] += G1 * RTC16
            E7[source] += G1 * RTC17
        RTC45, RTC46, RTC47 = normalize_arrays([normal_positive(mis.loc[i, 'TC45'], 0.16), normal_positive(mis.loc[i, 'TC46'], 0.16), normal_positive(mis.loc[i, 'TC47'], 0.16)])
        E5['Mismanagement'] += initial['Mismanagement'] * RTC45
        E6['Mismanagement'] += initial['Mismanagement'] * RTC46
        E7['Mismanagement'] += initial['Mismanagement'] * RTC47
        TC517 = normal_fraction(landfill.loc[i, 'TC517'], 0.05)
        TC616 = normal_fraction(landfill.loc[i, 'TC616'], 0.05)
        RTC816, RTC817 = normalize_arrays([normal_positive(inc.loc[i, 'TC816'], 0.25), normal_positive(inc.loc[i, 'TC817'], 0.25)])
        E16 = {s: np.zeros(N) for s in SOURCES}
        E17 = {s: np.zeros(N) for s in SOURCES}
        for source in SOURCES:
            E17[source] += E5[source] * TC517
            E16[source] += E6[source] * TC616
            E16[source] += E8[source] * RTC816
            E17[source] += E8[source] * RTC817
        RTC79, RTC710, RTC711, RTC712 = normalize_arrays([normal_positive(landfill.loc[i, 'TC79'], 0.16), normal_positive(landfill.loc[i, 'TC710'], 0.16), normal_positive(landfill.loc[i, 'TC711'], 0.16), normal_positive(landfill.loc[i, 'TC712'], 0.16)])
        TC916 = normal_fraction(landfill.loc[i, 'TC916'], 0.05)
        RTC1013, RTC1016 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1013'], 0.16), normal_positive(landfill.loc[i, 'TC1016'], 0.16)])
        RTC1113, RTC1116 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1113'], 0.16), normal_positive(landfill.loc[i, 'TC1116'], 0.16)])
        RTC1213, RTC1216 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1213'], 0.16), normal_positive(landfill.loc[i, 'TC1216'], 0.16)])
        RTC1314, RTC1315 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1314'], 0.16), normal_positive(landfill.loc[i, 'TC1315'], 0.16)])
        TC1417 = normal_fraction(landfill.loc[i, 'TC1417'], 0.05)
        for source in SOURCES:
            E9 = E7[source] * RTC79
            E10 = E7[source] * RTC710
            E11 = E7[source] * RTC711
            E12 = E7[source] * RTC712
            E16[source] += E9 * TC916
            E13 = E10 * RTC1013 + E11 * RTC1113 + E12 * RTC1213
            E16[source] += E10 * RTC1016 + E11 * RTC1116 + E12 * RTC1216
            E14 = E13 * RTC1314
            E17[source] += E14 * TC1417
        RTC1619, RTC1620 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1619'], 0.42), normal_positive(landfill.loc[i, 'TC1620'], 0.42)])
        RTC1718, RTC1719 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1718'], 0.46), normal_positive(landfill.loc[i, 'TC1719'], 0.46)])
        RTC1920, RTC1921 = normalize_arrays([normal_positive(landfill.loc[i, 'TC1920'], 0.16), normal_positive(landfill.loc[i, 'TC1921'], 0.16)])
        soil, ocean, river = ({}, {}, {})
        for source in SOURCES:
            E19_aquatic = E16[source] * RTC1619
            E20_aquatic = E16[source] * RTC1620
            soil[source] = E17[source] * RTC1718
            E19_soil = E17[source] * RTC1719
            E19 = E19_aquatic + E19_soil
            ocean[source] = E20_aquatic + E19 * RTC1920
            river[source] = E19 * RTC1921
        total_soil = sum((soil[source] for source in SOURCES))
        total_ocean = sum((ocean[source] for source in SOURCES))
        total_river = sum((river[source] for source in SOURCES))
        values = {f'{source}_MP': initial[source] for source in SOURCES}
        for source in SOURCES:
            values[f'{source}_soil'] = soil[source]
            values[f'{source}_ocean'] = ocean[source]
            values[f'{source}_river'] = river[source]
            values[f'{source}_environment'] = soil[source] + ocean[source] + river[source]
        values['total_soil'] = total_soil
        values['total_ocean'] = total_ocean
        values['total_river'] = total_river
        values['total_environment'] = total_soil + total_ocean + total_river
        row_result = []
        for indicator in INDICATORS:
            row_result.extend(stat(values[indicator]))
        result.iloc[i] = row_result
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f'\rSimulation progress: {i + 1:,}/{n:,}', end='', flush=True)
    print()
    write_stat_sheets(result, landfill, mc_file)
    print(f'Monte Carlo results saved: {mc_file}')
    organize_statistics(mc_file, grouped_file, scheme_name)
    return {'scheme': scheme_name, 'monte_carlo_results': str(mc_file), 'statistical_summary': str(grouped_file)}
