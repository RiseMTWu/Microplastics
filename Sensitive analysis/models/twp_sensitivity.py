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
INDICATORS = ['TWP', 'vehicle_TWP', 'airplane_TWP', 'TWP_soil', 'TWP_ocean', 'TWP_river', 'TWP_environment']
SCHEMES = {'Scheme_1_Mild_tilt': {'input': BASE_DIR / 'Scheme_1_Mild_tilt/dataset/2019_TWP.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_1_Mild_tilt' / 'twp'}, 'Scheme_2_Moderate_tilt': {'input': BASE_DIR / 'Scheme_2_Moderate_tilt/dataset/2019_TWP.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_2_Moderate_tilt' / 'twp'}, 'Scheme_3_Strong_tilt': {'input': BASE_DIR / 'Scheme_3_Strong_tilt/dataset/2019_TWP.xlsx', 'output': BASE_DIR / 'results' / 'Scheme_3_Strong_tilt' / 'twp'}}

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

def read_excel_input(input_file):
    return pd.read_excel(input_file, sheet_name='Data', engine='openpyxl')

def build_result_table(data):
    columns = [f'{indicator}_{stat_name}' for indicator in INDICATORS for stat_name in STAT_NAMES]
    return pd.DataFrame(np.nan, index=range(len(data)), columns=columns)

def add_base_info(result, data):
    city_column = 'city' if 'city' in data.columns else 'City'
    result.insert(0, 'city', data[city_column])
    result.insert(1, 'Category', data['Category'] if 'Category' in data.columns else np.nan)
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

def vehicle_class(row, vehicle_column, road_letter, first_m):
    vehicle = normal_positive(row[vehicle_column], 0.25)
    urban = normal_positive(row[f'cityroad{road_letter}'], 0.28) * normal_positive(row[f'M{first_m}'], 0.26)
    rural = normal_positive(row[f'ruralroad{road_letter}'], 0.28) * normal_positive(row[f'M{first_m + 1}'], 0.26)
    highway = normal_positive(row[f'highway{road_letter}'], 0.28) * normal_positive(row[f'M{first_m + 2}'], 0.26)
    return vehicle * (urban + rural + highway)

def organize_all_statistics(mc_file, grouped_file, scheme_name):
    with pd.ExcelWriter(grouped_file, engine='openpyxl') as writer:
        for stat_name in STAT_NAMES:
            print(f'Organizing: {scheme_name}—{stat_name}')
            df = pd.read_excel(mc_file, sheet_name=stat_name)
            missing = [column for column in ['city', 'Category', 'level', *INDICATORS] if column not in df.columns]
            if missing:
                raise KeyError(f'{scheme_name} / {stat_name} missing fields: {missing}')
            city_order = df[['city']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'city_order'})
            df[INDICATORS] = df[INDICATORS].apply(pd.to_numeric, errors='coerce')
            if df[INDICATORS].isna().any().any():
                raise ValueError(f'{scheme_name} / {stat_name} contains invalid numeric values.')
            df[INDICATORS] = df[INDICATORS] / 1000
            level_group = df.groupby('level', as_index=False, dropna=False)[INDICATORS].sum().sort_values('level')
            city_group = df.groupby('city', as_index=False, dropna=False)[INDICATORS].sum()
            city_group = city_group.merge(city_order, on='city', how='left', validate='one_to_one')
            city_group = city_group.sort_values('city_order').drop(columns='city_order')
            city_type_group = df.groupby('Category', as_index=False, dropna=False)[INDICATORS].sum().sort_values('Category')
            city_level_group = df.groupby(['Category', 'level'], as_index=False, dropna=False)[INDICATORS].sum()
            city_level_group = city_level_group.sort_values(['Category', 'level'])
            original_total = df[INDICATORS].sum().to_numpy()
            for name, grouped in {'level': level_group, 'city': city_group, 'city_type': city_type_group, 'city_level': city_level_group}.items():
                if not np.allclose(grouped[INDICATORS].sum().to_numpy(), original_total, rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f'{scheme_name}—{stat_name}—{name} grouped sum check failed.')
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
    mc_file = output_dir / 'TWP_result_MC.xlsx'
    grouped_file = output_dir / 'TWP_all_stats_grouped_result.xlsx'
    print('\n' + '=' * 80)
    print(f'Processing: {scheme_name}')
    print(f'Input file: {input_file}')
    print(f'Output directory: {output_dir}')
    print('=' * 80)
    data = read_excel_input(input_file).reset_index(drop=True)
    print(f'Input dimensions: {data.shape}')
    result = build_result_table(data)
    for i in range(len(data)):
        row = data.iloc[i]
        passenger = vehicle_class(row, 'small1', 'A', 1) + vehicle_class(row, 'medium1', 'B', 4) + vehicle_class(row, 'big1', 'C', 7)
        truck = vehicle_class(row, 'small2', 'D', 10) + vehicle_class(row, 'medium2', 'E', 13) + vehicle_class(row, 'big2', 'F', 16)
        passenger_ratio = normal_fraction(row['Car'], 0.28)
        truck_ratio = normal_fraction(row['Truck'], 0.28)
        tire_ratio = normal_fraction(row['TCR'], 0.11)
        airplane = normal_positive(row['Airplaneinput'], 0.28)
        passenger_twp = passenger * tire_ratio * passenger_ratio * 1e-05
        truck_twp = truck * tire_ratio * truck_ratio * 1e-05
        vehicle_twp = passenger_twp + truck_twp
        total_twp = vehicle_twp + airplane
        RTC13, RTC14, RTC15 = normalize_arrays([normal_positive(row['TC13'], 0.16), normal_positive(row['TC14'], 0.16), normal_positive(row['TC15'], 0.16)])
        RTC23, RTC24, RTC25 = normalize_arrays([normal_positive(row['TC23'], 0.16), normal_positive(row['TC24'], 0.16), normal_positive(row['TC25'], 0.16)])
        RTC314 = normal_fraction(row['TC314'], 0.05)
        RTC413 = normal_fraction(row['TC413'], 0.05)
        RTC56, RTC57, RTC58, RTC59 = normalize_arrays([normal_positive(row['TC56'], 0.16), normal_positive(row['TC57'], 0.16), normal_positive(row['TC58'], 0.16), normal_positive(row['TC59'], 0.16)])
        RTC613 = normal_fraction(row['TC613'], 0.05)
        RTC710, RTC713 = normalize_arrays([normal_positive(row['TC710'], 0.16), normal_positive(row['TC713'], 0.16)])
        RTC810, RTC813 = normalize_arrays([normal_positive(row['TC810'], 0.16), normal_positive(row['TC813'], 0.16)])
        RTC910, RTC913 = normalize_arrays([normal_positive(row['TC910'], 0.16), normal_positive(row['TC913'], 0.16)])
        RTC1011, RTC1012 = normalize_arrays([normal_positive(row['TC1011'], 0.16), normal_positive(row['TC1012'], 0.16)])
        RTC1114 = normal_fraction(row['TC1114'], 0.05)
        RTC1315, RTC1316 = normalize_arrays([normal_positive(row['TC1315'], 0.42), normal_positive(row['TC1316'], 0.42)])
        RTC1415, RTC1417 = normalize_arrays([normal_positive(row['TC1415'], 0.46), normal_positive(row['TC1417'], 0.46)])
        RTC1516, RTC1518 = normalize_arrays([normal_positive(row['TC1516'], 0.16), normal_positive(row['TC1518'], 0.16)])
        BOX3 = RTC13 * vehicle_twp + RTC23 * airplane
        BOX4 = RTC14 * vehicle_twp + RTC24 * airplane
        BOX5 = RTC15 * vehicle_twp + RTC25 * airplane
        BOX6, BOX7 = (BOX5 * RTC56, BOX5 * RTC57)
        BOX8, BOX9 = (BOX5 * RTC58, BOX5 * RTC59)
        BOX10 = BOX7 * RTC710 + BOX8 * RTC810 + BOX9 * RTC910
        BOX11, BOX12 = (BOX10 * RTC1011, BOX10 * RTC1012)
        BOX13 = BOX7 * RTC713 + BOX8 * RTC813 + BOX9 * RTC913 + BOX6 * RTC613 + BOX4 * RTC413
        BOX14 = BOX11 * RTC1114 + BOX3 * RTC314
        BOX15 = BOX13 * RTC1315 + BOX14 * RTC1415
        BOX16 = BOX13 * RTC1316 + BOX15 * RTC1516
        BOX17 = BOX14 * RTC1417
        BOX18 = BOX15 * RTC1518
        values = {'TWP': total_twp, 'vehicle_TWP': vehicle_twp, 'airplane_TWP': airplane, 'TWP_soil': BOX17, 'TWP_ocean': BOX16, 'TWP_river': BOX18, 'TWP_environment': BOX16 + BOX17 + BOX18}
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
    organize_all_statistics(mc_file, grouped_file, scheme_name)
    return {'scheme': scheme_name, 'monte_carlo_results': str(mc_file), 'all_statistics_summary': str(grouped_file)}
