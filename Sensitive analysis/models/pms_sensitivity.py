import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
N = int(os.environ.get('MC_DRAWS', '10000'))
MIN_VALUE = 0.0001
STAT_NAMES = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
INDICATORS = ['PMs_toothpaste', 'PMs_body_wash', 'PMs_facial_scrub', 'PMs_facial_cleanser', 'Input', 'D1_rinsed_off', 'D13_residual_soil', 'D15_ocean', 'D16_residual_river', 'total_environment']
SCHEMES = {'Scheme_1_Mild_tilt': {'input': BASE_DIR / 'Scheme_1_Mild_tilt/dataset/2019_PMs.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_1_Mild_tilt' / 'pms'}, 'Scheme_2_Moderate_tilt': {'input': BASE_DIR / 'Scheme_2_Moderate_tilt/dataset/2019_PMs.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_2_Moderate_tilt' / 'pms'}, 'Scheme_3_Strong_tilt': {'input': BASE_DIR / 'Scheme_3_Strong_tilt/dataset/2019_PMs.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_3_Strong_tilt' / 'pms'}}

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
    city_column = 'city' if 'city' in data.columns else 'City'
    result.insert(0, 'city', data[city_column])
    result.insert(1, 'Category', data['Category'] if 'Category' in data.columns else data['Category'] if 'Category' in data.columns else np.nan)
    result.insert(2, 'level', data['level'] if 'level' in data.columns else np.nan)
    result.insert(3, 'province', data['province'] if 'province' in data.columns else np.nan)
    return result

def write_stat_sheets(result, output_file):
    base = result[['city', 'Category', 'level', 'province']]
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for stat_name in STAT_NAMES:
            columns = [column for column in result.columns if column.endswith('_' + stat_name)]
            sheet = pd.concat([base, result[columns]], axis=1)
            sheet = sheet.rename(columns={column: column.removesuffix('_' + stat_name) for column in columns})
            sheet.to_excel(writer, sheet_name=stat_name, index=False)

def organize_all_statistics(mc_file, grouped_file, scheme_name):
    with pd.ExcelWriter(grouped_file, engine='openpyxl') as writer:
        for stat_name in STAT_NAMES:
            print(f'Organizing: {scheme_name}—{stat_name}')
            df = pd.read_excel(mc_file, sheet_name=stat_name)
            missing = [column for column in ['city', 'Category', 'level', *INDICATORS] if column not in df.columns]
            if missing:
                raise KeyError(f'{scheme_name} / {stat_name} worksheet missing fields: {missing}')
            city_order = df[['city']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'city_order'})
            df[INDICATORS] = df[INDICATORS].apply(pd.to_numeric, errors='coerce')
            if df[INDICATORS].isna().any().any():
                raise ValueError(f'{scheme_name} / {stat_name} worksheet contains invalid numeric values.')
            df[INDICATORS] = df[INDICATORS] / 1000
            level_group = df.groupby('level', as_index=False, dropna=False)[INDICATORS].sum().sort_values('level')
            city_group = df.groupby('city', as_index=False, dropna=False)[INDICATORS].sum()
            city_group = city_group.merge(city_order, on='city', how='left', validate='one_to_one')
            city_group = city_group.sort_values('city_order').drop(columns='city_order')
            city_type_group = df.groupby('Category', as_index=False, dropna=False)[INDICATORS].sum().sort_values('Category')
            city_level_group = df.groupby(['Category', 'level'], as_index=False, dropna=False)[INDICATORS].sum()
            city_level_group = city_level_group.sort_values(['Category', 'level'])
            original_total = df[INDICATORS].sum().to_numpy()
            grouped_tables = {'level': level_group, 'city': city_group, 'city_type': city_type_group, 'city_level': city_level_group}
            for group_name, grouped in grouped_tables.items():
                grouped_total = grouped[INDICATORS].sum().to_numpy()
                if not np.allclose(grouped_total, original_total, rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f'{scheme_name}—{stat_name}—{group_name} grouped sum check failed.')
            level_group.to_excel(writer, sheet_name=f'{stat_name}_level', index=False)
            city_group.to_excel(writer, sheet_name=f'{stat_name}_city', index=False)
            city_type_group.to_excel(writer, sheet_name=f'{stat_name}_city_type', index=False)
            city_level_group.to_excel(writer, sheet_name=f'{stat_name}_city_level', index=False)
    print(f'All statistical summaries saved: {grouped_file}')

def run_scheme(scheme_name, input_file, output_dir):
    input_file, output_dir = (Path(input_file), Path(output_dir))
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    output_dir.mkdir(parents=True, exist_ok=True)
    mc_file = output_dir / 'PMs_result_MC.xlsx'
    grouped_file = output_dir / 'PMs_all_stats_grouped_result.xlsx'
    print('\n' + '=' * 80)
    print(f'Processing: {scheme_name}')
    print(f'Input file: {input_file}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    df = pd.read_excel(input_file).reset_index(drop=True)
    print(f'Input dimensions: {df.shape}')
    result = build_result_table(df)
    for i in range(len(df)):
        row = df.iloc[i]
        Consumption_toothpaste = normal_positive(row['Consumption-toothpaste'], 0.15)
        Consumption_body_wash = normal_positive(row['Consumption-body_wash'], 0.15)
        Consumption_facial_scrub = normal_positive(row['Consumption-facial_scrub'], 0.15)
        Consumption_facial_cleanser = normal_positive(row['Consumption-facial_cleanser'], 0.15)
        PCPpercent = normal_fraction(row['PCPpercent'], 0.15)
        PPUW_toothpaste = normal_positive(row['PPUW-toothpaste'], 0.15)
        PPUW_body_wash = normal_positive(row['PPUW-body_wash'], 0.15)
        PPUW_facial_scrub = normal_positive(row['PPUW-facial_scrub'], 0.15)
        PPUW_facial_cleanser = normal_positive(row['PPUW-facial_cleanser'], 0.15)
        PMs_toothpaste_percent = normal_fraction(row['PMs-toothpaste'], 0.13)
        PMs_body_wash_percent = normal_fraction(row['PMs-body_wash'], 0.13)
        PMs_facial_scrub_percent = normal_fraction(row['PMs-facial_scrub'], 0.13)
        PMs_facial_cleanser_percent = normal_fraction(row['PMs-facial_cleanser'], 0.13)
        EF_toothpaste = normal_positive(row['EF-toothpaste'], 0.13)
        EF_body_wash = normal_positive(row['EF-body_wash'], 0.13)
        EF_facial_scrub = normal_positive(row['EF-facial_scrub'], 0.13)
        EF_facial_cleanser = normal_positive(row['EF-facial_cleanser'], 0.13)
        PMs_toothpaste = Consumption_toothpaste * PCPpercent / PPUW_toothpaste * PMs_toothpaste_percent * EF_toothpaste * 0.1
        PMs_body_wash = Consumption_body_wash * PCPpercent / PPUW_body_wash * PMs_body_wash_percent * EF_body_wash * 0.1
        PMs_facial_scrub = Consumption_facial_scrub * PCPpercent / PPUW_facial_scrub * PMs_facial_scrub_percent * EF_facial_scrub * 0.1
        PMs_facial_cleanser = Consumption_facial_cleanser * PCPpercent / PPUW_facial_cleanser * PMs_facial_cleanser_percent * EF_facial_cleanser * 0.1
        Input = PMs_toothpaste + PMs_body_wash + PMs_facial_scrub + PMs_facial_cleanser
        D1 = Input * 0.95
        RTC12, RTC13 = normalize_arrays([normal_positive(row['TC12'], 0.43), normal_positive(row['TC13'], 0.43)])
        D2, D3 = (D1 * RTC12, D1 * RTC13)
        D11_from_D2 = D2 * normal_fraction(row['TC211'], 0.05)
        RTC34, RTC35, RTC36, RTC37 = normalize_arrays([normal_positive(row['TC34'], 0.16), normal_positive(row['TC35'], 0.16), normal_positive(row['TC36'], 0.16), normal_positive(row['TC37'], 0.16)])
        D4, D5, D6, D7 = (D3 * RTC34, D3 * RTC35, D3 * RTC36, D3 * RTC37)
        D11_from_D4 = D4 * normal_fraction(row['TC411'], 0.05)
        RTC511, RTC58 = normalize_arrays([normal_positive(row['TC511'], 0.16), normal_positive(row['TC58'], 0.16)])
        D11_from_D5, D8_from_D5 = (D5 * RTC511, D5 * RTC58)
        RTC611, RTC68 = normalize_arrays([normal_positive(row['TC611'], 0.16), normal_positive(row['TC68'], 0.16)])
        D11_from_D6, D8_from_D6 = (D6 * RTC611, D6 * RTC68)
        RTC711, RTC78 = normalize_arrays([normal_positive(row['TC711'], 0.16), normal_positive(row['TC78'], 0.16)])
        D11_from_D7, D8_from_D7 = (D7 * RTC711, D7 * RTC78)
        D8 = D8_from_D5 + D8_from_D6 + D8_from_D7
        RTC89, RTC810 = normalize_arrays([normal_positive(row['TC89'], 0.16), normal_positive(row['TC810'], 0.16)])
        D9, D10 = (D8 * RTC89, D8 * RTC810)
        D12 = D9 * normal_fraction(row['TC912'], 0.05)
        D11 = D11_from_D2 + D11_from_D4 + D11_from_D5 + D11_from_D6 + D11_from_D7
        RTC1114, RTC1115 = normalize_arrays([normal_positive(row['TC1114'], 0.42), normal_positive(row['TC1115'], 0.42)])
        D14_from_D11, D15_from_D11 = (D11 * RTC1114, D11 * RTC1115)
        RTC1213, RTC1214 = normalize_arrays([normal_positive(row['TC1213'], 0.46), normal_positive(row['TC1214'], 0.46)])
        D13, D14_from_D12 = (D12 * RTC1213, D12 * RTC1214)
        D14 = D14_from_D11 + D14_from_D12
        RTC1415, RTC1416 = normalize_arrays([normal_positive(row['TC1415'], 0.16), normal_positive(row['TC1416'], 0.16)])
        D15 = D15_from_D11 + D14 * RTC1415
        D16 = D14 * RTC1416
        total_environment = D13 + D15 + D16
        values = {'PMs_toothpaste': PMs_toothpaste, 'PMs_body_wash': PMs_body_wash, 'PMs_facial_scrub': PMs_facial_scrub, 'PMs_facial_cleanser': PMs_facial_cleanser, 'Input': Input, 'D1_rinsed_off': D1, 'D13_residual_soil': D13, 'D15_ocean': D15, 'D16_residual_river': D16, 'total_environment': total_environment}
        row_result = []
        for indicator in INDICATORS:
            row_result.extend(stat(values[indicator]))
        result.iloc[i] = row_result
        if (i + 1) % 100 == 0 or i + 1 == len(df):
            print(f'\rSimulation progress: {i + 1:,}/{len(df):,}', end='', flush=True)
    print()
    result = add_base_info(result, df)
    write_stat_sheets(result, mc_file)
    print(f'PMs Monte Carlo results saved: {mc_file}')
    organize_all_statistics(mc_file=mc_file, grouped_file=grouped_file, scheme_name=scheme_name)
    return {'scheme': scheme_name, 'monte_carlo_results': str(mc_file), 'grouped_results': str(grouped_file)}
