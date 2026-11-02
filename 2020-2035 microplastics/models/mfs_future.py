import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
dataset_dir = BASE_DIR / 'MFs-dataset'
output_dir = BASE_DIR / 'results' / 'MFs'
output_dir.mkdir(parents=True, exist_ok=True)
years = list(range(2020, 2036))
scenarios = ['BAU', 'Mitigation']
N = int(os.environ.get('MC_DRAWS', '10000'))
MIN_VALUE = 0.0001
BASE_RANDOM_SEED = 42
SKIP_EXISTING = True
stat_names = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
indicator_columns = ['MFs', 'machine_MFs', 'hand_MFs', 'MFs_soil', 'MFs_ocean', 'MFs_river', 'MFs_environment']

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def normal_positive(mean, cv, size=N):
    mean = pd.to_numeric(mean, errors='coerce')
    mean = 0 if pd.isna(mean) else float(mean)
    x = np.random.normal(loc=mean, scale=abs(cv * mean), size=size)
    x[x < 0] = MIN_VALUE
    return x

def normal_fraction(mean, cv, size=N):
    x = normal_positive(mean, cv, size)
    return np.clip(x, 0, 1)

def normalize_arrays(arrays):
    total = np.zeros(N)
    for a in arrays:
        total += a
    total[total <= 0] = MIN_VALUE
    return [a / total for a in arrays]

def stat(x):
    out = [np.mean(x), np.std(x)]
    out.extend(np.quantile(x, [0.05, 0.25, 0.75, 0.95]))
    return out

def build_result_table(data_input):
    result_columns = []
    for indicator in indicator_columns:
        for stat_name in stat_names:
            result_columns.append(f'{indicator}_{stat_name}')
    return pd.DataFrame(np.nan, index=range(len(data_input)), columns=result_columns)

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{label} missing required columns: {missing}; available columns: {list(df.columns)}')

def find_population_col(df, year):
    if 'population' not in df.columns:
        raise ValueError(f'Population column not found for {year}; available columns: {list(df.columns)}')
    return 'population'

def add_base_info(result_df, data_input):
    result_df.insert(0, 'city', data_input['city'] if 'city' in data_input.columns else data_input['City'])
    result_df.insert(1, 'Category', data_input['Category'] if 'Category' in data_input.columns else np.nan)
    result_df.insert(2, 'level', data_input['level'] if 'level' in data_input.columns else np.nan)
    result_df.insert(3, 'province', data_input['province'] if 'province' in data_input.columns else np.nan)
    return result_df

def write_stat_sheets(result_df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    base_info = result_df[['city', 'Category', 'level', 'province']]
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for stat_name in stat_names:
            stat_cols = [col for col in result_df.columns if col.endswith('_' + stat_name)]
            sheet_df = pd.concat([base_info, result_df[stat_cols]], axis=1)
            sheet_df = sheet_df.rename(columns={col: col.replace('_' + stat_name, '') for col in stat_cols})
            sheet_df.to_excel(writer, sheet_name=stat_name, index=False)

def calc_seed(year, scenario):
    scenario_offset = 0 if scenario == 'BAU' else 100000
    return BASE_RANDOM_SEED + (year - 2020) * 1000 + scenario_offset

def output_file_for(year, scenario):
    return output_dir / f'{year}{scenario}-MFs.xlsx'

def read_year_data(scenario, year):
    input_file = dataset_dir / f'{year}-{scenario}.xlsx'
    if not input_file.exists():
        raise FileNotFoundError(f'Input file not found: {input_file}')
    return clean_columns(pd.read_excel(input_file, sheet_name='Data', engine='openpyxl'))

def run_one_year_scenario(year, scenario):
    np.random.seed(calc_seed(year, scenario))
    data_input = read_year_data(scenario, year)
    print(f'{year}-{scenario} input dimensions: {data_input.shape}')
    population_col = find_population_col(data_input, year)
    required_cols = ['city', 'Category', 'level', population_col, 'synthetic fibers', 'FrequencyMW', 'FrequencyHW', 'EF-MW', 'EF-HW', 'Load-MW', 'Load-HW', 'TC14', 'TC15', 'TC23', 'TC24', 'TC25', 'TC314', 'TC413', 'TC56', 'TC57', 'TC58', 'TC59', 'TC613', 'TC710', 'TC713', 'TC810', 'TC813', 'TC910', 'TC913', 'TC1011', 'TC1012', 'TC1114', 'TC1315', 'TC1316', 'TC1415', 'TC1417', 'TC1516', 'TC1518']
    require_columns(data_input, required_cols, f'{year}-{scenario}')
    city_result = build_result_table(data_input)
    micro_columns = ['total', 'BOX1', 'BOX2', 'BOX3', 'BOX4', 'BOX5', 'BOX6', 'BOX7', 'BOX8', 'BOX9', 'BOX10', 'BOX11', 'BOX12', 'BOX13', 'BOX14', 'BOX15', 'BOX16', 'BOX17', 'BOX18']
    micro = pd.DataFrame(np.nan, index=range(N), columns=micro_columns)
    for i in range(len(data_input)):
        population = normal_positive(data_input.loc[i, population_col], 0.42)
        synthetic_fiber_ratio = normal_fraction(data_input.loc[i, 'synthetic fibers'], 0.15)
        machine_wash_freq = normal_positive(data_input.loc[i, 'FrequencyMW'], 0.43)
        hand_wash_freq = normal_positive(data_input.loc[i, 'FrequencyHW'], 0.43)
        machine_ef = normal_positive(data_input.loc[i, 'EF-MW'], 0.17)
        hand_ef = normal_positive(data_input.loc[i, 'EF-HW'], 0.17)
        machine_load = normal_positive(data_input.loc[i, 'Load-MW'], 0.42)
        hand_load = normal_positive(data_input.loc[i, 'Load-HW'], 0.42)
        TC14 = normal_positive(data_input.loc[i, 'TC14'], 0.43)
        TC15 = normal_positive(data_input.loc[i, 'TC15'], 0.43)
        RTC14, RTC15 = normalize_arrays([TC14, TC15])
        TC23 = normal_positive(data_input.loc[i, 'TC23'], 0.43)
        TC24 = normal_positive(data_input.loc[i, 'TC24'], 0.43)
        TC25 = normal_positive(data_input.loc[i, 'TC25'], 0.43)
        RTC23, RTC24, RTC25 = normalize_arrays([TC23, TC24, TC25])
        RTC314 = normal_fraction(data_input.loc[i, 'TC314'], 0.05)
        RTC413 = normal_fraction(data_input.loc[i, 'TC413'], 0.05)
        TC56 = normal_positive(data_input.loc[i, 'TC56'], 0.16)
        TC57 = normal_positive(data_input.loc[i, 'TC57'], 0.16)
        TC58 = normal_positive(data_input.loc[i, 'TC58'], 0.16)
        TC59 = normal_positive(data_input.loc[i, 'TC59'], 0.16)
        RTC56, RTC57, RTC58, RTC59 = normalize_arrays([TC56, TC57, TC58, TC59])
        RTC613 = normal_fraction(data_input.loc[i, 'TC613'], 0.05)
        TC710 = normal_positive(data_input.loc[i, 'TC710'], 0.16)
        TC713 = normal_positive(data_input.loc[i, 'TC713'], 0.16)
        RTC710, RTC713 = normalize_arrays([TC710, TC713])
        TC810 = normal_positive(data_input.loc[i, 'TC810'], 0.16)
        TC813 = normal_positive(data_input.loc[i, 'TC813'], 0.16)
        RTC810, RTC813 = normalize_arrays([TC810, TC813])
        TC910 = normal_positive(data_input.loc[i, 'TC910'], 0.16)
        TC913 = normal_positive(data_input.loc[i, 'TC913'], 0.16)
        RTC910, RTC913 = normalize_arrays([TC910, TC913])
        TC1011 = normal_positive(data_input.loc[i, 'TC1011'], 0.16)
        TC1012 = normal_positive(data_input.loc[i, 'TC1012'], 0.16)
        RTC1011, RTC1012 = normalize_arrays([TC1011, TC1012])
        RTC1114 = normal_fraction(data_input.loc[i, 'TC1114'], 0.05)
        TC1315 = normal_positive(data_input.loc[i, 'TC1315'], 0.42)
        TC1316 = normal_positive(data_input.loc[i, 'TC1316'], 0.42)
        RTC1315, RTC1316 = normalize_arrays([TC1315, TC1316])
        TC1415 = normal_positive(data_input.loc[i, 'TC1415'], 0.46)
        TC1417 = normal_positive(data_input.loc[i, 'TC1417'], 0.46)
        RTC1415, RTC1417 = normalize_arrays([TC1415, TC1417])
        TC1516 = normal_positive(data_input.loc[i, 'TC1516'], 0.16)
        TC1518 = normal_positive(data_input.loc[i, 'TC1518'], 0.16)
        RTC1516, RTC1518 = normalize_arrays([TC1516, TC1518])
        micro['BOX1'] = population * synthetic_fiber_ratio * machine_wash_freq * machine_load * machine_ef * 1e-08
        micro['BOX2'] = population * synthetic_fiber_ratio * hand_wash_freq * hand_load * hand_ef * 1e-08
        micro['total'] = micro['BOX1'] + micro['BOX2']
        micro['BOX3'] = RTC23 * micro['BOX2']
        micro['BOX4'] = RTC14 * micro['BOX1'] + RTC24 * micro['BOX2']
        micro['BOX5'] = RTC15 * micro['BOX1'] + RTC25 * micro['BOX2']
        micro['BOX6'] = micro['BOX5'] * RTC56
        micro['BOX7'] = micro['BOX5'] * RTC57
        micro['BOX8'] = micro['BOX5'] * RTC58
        micro['BOX9'] = micro['BOX5'] * RTC59
        micro['BOX10'] = micro['BOX7'] * RTC710 + micro['BOX8'] * RTC810 + micro['BOX9'] * RTC910
        micro['BOX11'] = micro['BOX10'] * RTC1011
        micro['BOX12'] = micro['BOX10'] * RTC1012
        micro['BOX13'] = micro['BOX4'] * RTC413 + micro['BOX6'] * RTC613 + micro['BOX7'] * RTC713 + micro['BOX8'] * RTC813 + micro['BOX9'] * RTC913
        micro['BOX14'] = micro['BOX3'] * RTC314 + micro['BOX11'] * RTC1114
        micro['BOX15'] = micro['BOX13'] * RTC1315 + micro['BOX14'] * RTC1415
        micro['BOX16'] = micro['BOX13'] * RTC1316 + micro['BOX15'] * RTC1516
        micro['BOX17'] = micro['BOX14'] * RTC1417
        micro['BOX18'] = micro['BOX15'] * RTC1518
        values_dict = {'MFs': micro['total'], 'machine_MFs': micro['BOX1'], 'hand_MFs': micro['BOX2'], 'MFs_soil': micro['BOX17'], 'MFs_ocean': micro['BOX16'], 'MFs_river': micro['BOX18'], 'MFs_environment': micro['BOX16'] + micro['BOX17'] + micro['BOX18']}
        row_result = []
        for indicator in indicator_columns:
            row_result.extend(stat(values_dict[indicator]))
        city_result.iloc[i] = row_result
        if (i + 1) % 200 == 0 or i + 1 == len(data_input):
            print(f'  {year}-{scenario}: completed {i + 1}/{len(data_input)} rows')
    city_result = add_base_info(city_result, data_input)
    output_file = output_file_for(year, scenario)
    write_stat_sheets(city_result, output_file)
    print(f'Completed: {output_file}')

def main():
    summary_rows = []
    for year in years:
        for scenario in scenarios:
            output_file = output_file_for(year, scenario)
            if SKIP_EXISTING and output_file.exists():
                print(f'Skipping existing results: {output_file}')
                summary_rows.append({'year': year, 'scenario': scenario, 'status': 'skipped_existing', 'output_file': str(output_file)})
                continue
            print('=' * 80)
            print(f'Starting: {year}-{scenario}')
            run_one_year_scenario(year, scenario)
            summary_rows.append({'year': year, 'scenario': scenario, 'status': 'completed', 'output_file': str(output_file), 'N': N, 'random_seed': calc_seed(year, scenario)})
    summary_file = output_dir / 'MFs_MC_batch_summary.xlsx'
    pd.DataFrame(summary_rows).to_excel(summary_file, index=False, engine='openpyxl')
    print('=' * 80)
    print(f'Completed; batch summary: {summary_file}')
if __name__ == '__main__':
    main()
