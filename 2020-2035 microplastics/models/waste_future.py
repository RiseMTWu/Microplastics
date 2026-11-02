import os
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
dataset_dir = BASE_DIR / 'Waste-dataset'
output_dir = BASE_DIR / 'results' / 'Waste'
joint_output_dir = output_dir / 'city_MC'
grouped_output_dir = output_dir / 'grouped_means'
joint_output_dir.mkdir(parents=True, exist_ok=True)
grouped_output_dir.mkdir(parents=True, exist_ok=True)
YEARS = list(range(2020, 2036))
SCENARIOS = ['BAU', 'Mitigation']
N = int(os.environ.get('MC_DRAWS', '10000'))
MIN_VALUE = 0.0001
BASE_RANDOM_SEED = 42
SKIP_EXISTING = True
sources = ['Landfill', 'Incineration', 'Recycling', 'Mismanagement']
fates = ['soil', 'ocean', 'river']
stat_names = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
base_cols = ['city', 'Category', 'level', 'province']
indicator_columns = []
for s in sources:
    indicator_columns.append(f'{s}_MP')
for s in sources:
    for f in fates:
        indicator_columns.append(f'{s}_{f}')
for s in sources:
    indicator_columns.append(f'{s}_environment')
for f in fates:
    indicator_columns.append(f'total_{f}')
indicator_columns.append('total_environment')
result_columns = []
for indicator in indicator_columns:
    for stat_name in stat_names:
        result_columns.append(f'{indicator}_{stat_name}')
value_cols_for_group = ['Landfill_MP', 'Incineration_MP', 'Recycling_MP', 'Mismanagement_MP', 'Landfill_soil', 'Landfill_ocean', 'Landfill_river', 'Incineration_soil', 'Incineration_ocean', 'Incineration_river', 'Recycling_soil', 'Recycling_ocean', 'Recycling_river', 'Mismanagement_soil', 'Mismanagement_ocean', 'Mismanagement_river', 'Landfill_environment', 'Incineration_environment', 'Recycling_environment', 'Mismanagement_environment', 'total_soil', 'total_ocean', 'total_river', 'total_environment']

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

def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'{label} missing columns: {missing}; available columns: {list(df.columns)}')

def make_dataset_path(year, scenario):
    return dataset_dir / f'{year}-{scenario}.xlsx'

def make_outputs(year, scenario):
    tag = f'{year}-{scenario}'
    joint_file = joint_output_dir / f'{tag}_waste_env_joint_result_MC.xlsx'
    grouped_file = grouped_output_dir / f'{tag}_waste_env_mean_grouped_result.xlsx'
    return (joint_file, grouped_file)

def calc_seed(year, scenario):
    scenario_offset = 0 if scenario == 'BAU' else 100000
    return BASE_RANDOM_SEED + (year - 2020) * 1000 + scenario_offset

def read_dataset(input_file):
    df_landfill = clean_columns(pd.read_excel(input_file, sheet_name='Landfill', engine='openpyxl'))
    df_inc = clean_columns(pd.read_excel(input_file, sheet_name='Incineration', engine='openpyxl'))
    df_rec = clean_columns(pd.read_excel(input_file, sheet_name='Recycling', engine='openpyxl'))
    df_mis = clean_columns(pd.read_excel(input_file, sheet_name='Mismanagement', engine='openpyxl'))
    require_columns(df_landfill, base_cols + ['Landfill/kt', 'Landfill-EF'], 'Landfill')
    require_columns(df_inc, base_cols + ['Incineration/kt', 'Incineration-EF'], 'Incineration')
    require_columns(df_rec, base_cols + ['Recycling/kt', 'Recycling-EF'], 'Recycling')
    require_columns(df_mis, base_cols + ['Mismanagement/kt', 'Mismanagement-EF'], 'Mismanagement')
    n = len(df_landfill)
    for name, df in [('Incineration', df_inc), ('Recycling', df_rec), ('Mismanagement', df_mis)]:
        if len(df) != n:
            raise ValueError(f'{input_file.name} in {name} row count does not match Landfill.')
    return (df_landfill, df_inc, df_rec, df_mis)

def run_one_dataset(input_file, joint_output_file, grouped_output_file, random_seed):
    np.random.seed(random_seed)
    df_landfill, df_inc, df_rec, df_mis = read_dataset(input_file)
    n = len(df_landfill)
    city_result = pd.DataFrame(np.nan, index=range(n), columns=result_columns)
    for i in range(n):
        E1_direct = normal_positive(df_landfill.loc[i, 'Landfill/kt'], 0.26) * normal_positive(df_landfill.loc[i, 'Landfill-EF'], 0.45)
        E2_direct = normal_positive(df_inc.loc[i, 'Incineration/kt'], 0.26) * normal_positive(df_inc.loc[i, 'Incineration-EF'], 0.45)
        E3_direct = normal_positive(df_rec.loc[i, 'Recycling/kt'], 0.26) * normal_positive(df_rec.loc[i, 'Recycling-EF'], 0.45)
        E4_direct = normal_positive(df_mis.loc[i, 'Mismanagement/kt'], 0.26) * normal_positive(df_mis.loc[i, 'Mismanagement-EF'], 0.45)
        initial_MP = {'Landfill': E1_direct, 'Incineration': E2_direct, 'Recycling': E3_direct, 'Mismanagement': E4_direct}
        landfill_input = {s: np.zeros(N) for s in sources}
        recycle_input = {s: np.zeros(N) for s in sources}
        E5 = {s: np.zeros(N) for s in sources}
        E6 = {s: np.zeros(N) for s in sources}
        E7 = {s: np.zeros(N) for s in sources}
        E8 = {s: np.zeros(N) for s in sources}
        landfill_input['Landfill'] += E1_direct
        recycle_input['Recycling'] += E3_direct
        EG221 = normal_positive(df_inc.loc[i, 'EG221'], 0.43)
        EG222 = normal_positive(df_inc.loc[i, 'EG222'], 0.43)
        REG221, REG222 = normalize_arrays([EG221, EG222])
        G21_bottom_ash = E2_direct * REG221
        G22_flue_gas = E2_direct * REG222
        GE211 = normal_positive(df_inc.loc[i, 'GE211'], 0.43)
        GE213 = normal_positive(df_inc.loc[i, 'GE213'], 0.43)
        RGE211, RGE213 = normalize_arrays([GE211, GE213])
        landfill_input['Incineration'] += G21_bottom_ash * RGE211
        recycle_input['Incineration'] += G21_bottom_ash * RGE213
        GG2223 = normal_positive(df_inc.loc[i, 'GG2223'], 0.43)
        GG2224 = normal_positive(df_inc.loc[i, 'GG2224'], 0.43)
        TC28 = normal_positive(df_inc.loc[i, 'TC28'], 0.43)
        RGG2223, RGG2224, RTC28 = normalize_arrays([GG2223, GG2224, TC28])
        G23_fly_ash = G22_flue_gas * RGG2223
        G24_washing_water_inc = G22_flue_gas * RGG2224
        E8['Incineration'] += G22_flue_gas * RTC28
        GE231 = normal_positive(df_inc.loc[i, 'GE231'], 0.43)
        GE233 = normal_positive(df_inc.loc[i, 'GE233'], 0.43)
        RGE231, RGE233 = normalize_arrays([GE231, GE233])
        landfill_input['Incineration'] += G23_fly_ash * RGE231
        recycle_input['Incineration'] += G23_fly_ash * RGE233
        TC27 = normal_fraction(df_inc.loc[i, 'TC27'], 0.05)
        E7['Incineration'] += G24_washing_water_inc * TC27
        EG331 = normal_positive(df_rec.loc[i, 'EG331'], 0.43)
        EG332 = normal_positive(df_rec.loc[i, 'EG332'], 0.43)
        TC38 = normal_positive(df_rec.loc[i, 'TC38'], 0.43)
        REG331, REG332, RTC38 = normalize_arrays([EG331, EG332, TC38])
        GE311 = normal_fraction(df_rec.loc[i, 'GE311'], 0.05)
        TC35 = normal_positive(df_rec.loc[i, 'TC35'], 0.26)
        TC36 = normal_positive(df_rec.loc[i, 'TC36'], 0.25)
        TC37 = normal_positive(df_rec.loc[i, 'TC37'], 0.43)
        RTC35, RTC36, RTC37 = normalize_arrays([TC35, TC36, TC37])
        for s in sources:
            G31_solid_residues = recycle_input[s] * REG331
            G32_washing_water_rec = recycle_input[s] * REG332
            E8[s] += recycle_input[s] * RTC38
            landfill_input[s] += G31_solid_residues * GE311
            E5[s] += G32_washing_water_rec * RTC35
            E6[s] += G32_washing_water_rec * RTC36
            E7[s] += G32_washing_water_rec * RTC37
        EG11 = normal_fraction(df_landfill.loc[i, 'EG11'], 0.25)
        TC15 = normal_positive(df_landfill.loc[i, 'TC15'], 0.26)
        TC16 = normal_positive(df_landfill.loc[i, 'TC16'], 0.25)
        TC17 = normal_positive(df_landfill.loc[i, 'TC17'], 0.43)
        RTC15, RTC16, RTC17 = normalize_arrays([TC15, TC16, TC17])
        for s in sources:
            G1_leachate = landfill_input[s] * EG11
            E5[s] += G1_leachate * RTC15
            E6[s] += G1_leachate * RTC16
            E7[s] += G1_leachate * RTC17
        TC45 = normal_positive(df_mis.loc[i, 'TC45'], 0.16)
        TC46 = normal_positive(df_mis.loc[i, 'TC46'], 0.16)
        TC47 = normal_positive(df_mis.loc[i, 'TC47'], 0.16)
        RTC45, RTC46, RTC47 = normalize_arrays([TC45, TC46, TC47])
        E5['Mismanagement'] += E4_direct * RTC45
        E6['Mismanagement'] += E4_direct * RTC46
        E7['Mismanagement'] += E4_direct * RTC47
        TC517 = normal_fraction(df_landfill.loc[i, 'TC517'], 0.05)
        TC616 = normal_fraction(df_landfill.loc[i, 'TC616'], 0.05)
        TC816 = normal_positive(df_inc.loc[i, 'TC816'], 0.25)
        TC817 = normal_positive(df_inc.loc[i, 'TC817'], 0.25)
        RTC816, RTC817 = normalize_arrays([TC816, TC817])
        E16 = {s: np.zeros(N) for s in sources}
        E17 = {s: np.zeros(N) for s in sources}
        for s in sources:
            E17[s] += E5[s] * TC517
            E16[s] += E6[s] * TC616
            E16[s] += E8[s] * RTC816
            E17[s] += E8[s] * RTC817
        TC79 = normal_positive(df_landfill.loc[i, 'TC79'], 0.16)
        TC710 = normal_positive(df_landfill.loc[i, 'TC710'], 0.16)
        TC711 = normal_positive(df_landfill.loc[i, 'TC711'], 0.16)
        TC712 = normal_positive(df_landfill.loc[i, 'TC712'], 0.16)
        RTC79, RTC710, RTC711, RTC712 = normalize_arrays([TC79, TC710, TC711, TC712])
        TC916 = normal_fraction(df_landfill.loc[i, 'TC916'], 0.05)
        TC1013 = normal_positive(df_landfill.loc[i, 'TC1013'], 0.16)
        TC1016 = normal_positive(df_landfill.loc[i, 'TC1016'], 0.16)
        RTC1013, RTC1016 = normalize_arrays([TC1013, TC1016])
        TC1113 = normal_positive(df_landfill.loc[i, 'TC1113'], 0.16)
        TC1116 = normal_positive(df_landfill.loc[i, 'TC1116'], 0.16)
        RTC1113, RTC1116 = normalize_arrays([TC1113, TC1116])
        TC1213 = normal_positive(df_landfill.loc[i, 'TC1213'], 0.16)
        TC1216 = normal_positive(df_landfill.loc[i, 'TC1216'], 0.16)
        RTC1213, RTC1216 = normalize_arrays([TC1213, TC1216])
        TC1314 = normal_positive(df_landfill.loc[i, 'TC1314'], 0.16)
        TC1315 = normal_positive(df_landfill.loc[i, 'TC1315'], 0.16)
        RTC1314, RTC1315 = normalize_arrays([TC1314, TC1315])
        TC1417 = normal_fraction(df_landfill.loc[i, 'TC1417'], 0.05)
        for s in sources:
            E9 = E7[s] * RTC79
            E10 = E7[s] * RTC710
            E11 = E7[s] * RTC711
            E12 = E7[s] * RTC712
            E16[s] += E9 * TC916
            E13_from_E10 = E10 * RTC1013
            E16[s] += E10 * RTC1016
            E13_from_E11 = E11 * RTC1113
            E16[s] += E11 * RTC1116
            E13_from_E12 = E12 * RTC1213
            E16[s] += E12 * RTC1216
            E13 = E13_from_E10 + E13_from_E11 + E13_from_E12
            E14 = E13 * RTC1314
            E17[s] += E14 * TC1417
        TC1619 = normal_positive(df_landfill.loc[i, 'TC1619'], 0.42)
        TC1620 = normal_positive(df_landfill.loc[i, 'TC1620'], 0.42)
        RTC1619, RTC1620 = normalize_arrays([TC1619, TC1620])
        TC1718 = normal_positive(df_landfill.loc[i, 'TC1718'], 0.46)
        TC1719 = normal_positive(df_landfill.loc[i, 'TC1719'], 0.46)
        RTC1718, RTC1719 = normalize_arrays([TC1718, TC1719])
        TC1920 = normal_positive(df_landfill.loc[i, 'TC1920'], 0.16)
        TC1921 = normal_positive(df_landfill.loc[i, 'TC1921'], 0.16)
        RTC1920, RTC1921 = normalize_arrays([TC1920, TC1921])
        E18_soil = {}
        E20_ocean = {}
        E21_river = {}
        for s in sources:
            E19_from_aquatic = E16[s] * RTC1619
            E20_from_aquatic = E16[s] * RTC1620
            E18_soil[s] = E17[s] * RTC1718
            E19_from_soil = E17[s] * RTC1719
            E19_river_intermediate = E19_from_aquatic + E19_from_soil
            E20_ocean[s] = E20_from_aquatic + E19_river_intermediate * RTC1920
            E21_river[s] = E19_river_intermediate * RTC1921
        source_environment = {}
        for s in sources:
            source_environment[s] = E18_soil[s] + E20_ocean[s] + E21_river[s]
        total_E18_soil = sum((E18_soil[s] for s in sources))
        total_E20_ocean = sum((E20_ocean[s] for s in sources))
        total_E21_river = sum((E21_river[s] for s in sources))
        total_environment = total_E18_soil + total_E20_ocean + total_E21_river
        values_dict = {}
        for s in sources:
            values_dict[f'{s}_MP'] = initial_MP[s]
        for s in sources:
            values_dict[f'{s}_soil'] = E18_soil[s]
            values_dict[f'{s}_ocean'] = E20_ocean[s]
            values_dict[f'{s}_river'] = E21_river[s]
            values_dict[f'{s}_environment'] = source_environment[s]
        values_dict['total_soil'] = total_E18_soil
        values_dict['total_ocean'] = total_E20_ocean
        values_dict['total_river'] = total_E21_river
        values_dict['total_environment'] = total_environment
        row_result = []
        for indicator in indicator_columns:
            row_result.extend(stat(values_dict[indicator]))
        city_result.iloc[i] = row_result
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f'    Completed {i + 1}/{n} rows')
    city_result.insert(0, 'city', df_landfill['city'])
    city_result.insert(1, 'Category', df_landfill['Category'])
    city_result.insert(2, 'level', df_landfill['level'])
    city_result.insert(3, 'province', df_landfill['province'])
    base_info = city_result[base_cols]
    with pd.ExcelWriter(joint_output_file, engine='openpyxl') as writer:
        for stat_name in stat_names:
            stat_cols = [col for col in city_result.columns if col.endswith('_' + stat_name)]
            sheet_df = pd.concat([base_info, city_result[stat_cols]], axis=1)
            rename_dict = {col: col.replace('_' + stat_name, '') for col in stat_cols}
            sheet_df = sheet_df.rename(columns=rename_dict)
            sheet_df.to_excel(writer, sheet_name=stat_name, index=False)
    build_grouped_result(joint_output_file, grouped_output_file)

def build_grouped_result(joint_output_file, grouped_output_file):
    df = pd.read_excel(joint_output_file, sheet_name='mean', engine='openpyxl')
    df = clean_columns(df)
    city_order = df[['city']].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index': 'city_order'})
    require_columns(df, ['city', 'Category', 'level'] + value_cols_for_group, 'mean_results')
    df[value_cols_for_group] = df[value_cols_for_group].apply(pd.to_numeric, errors='coerce')
    level_group = df.groupby('level', as_index=False)[value_cols_for_group].sum().sort_values('level')
    city_group = df.groupby('city', as_index=False)[value_cols_for_group].sum().merge(city_order, on='city', how='left').sort_values('city_order').drop(columns='city_order')
    category_group = df.groupby('Category', as_index=False)[value_cols_for_group].sum().sort_values('Category')
    category_level_group = df.groupby(['Category', 'level'], as_index=False)[value_cols_for_group].sum().sort_values(['Category', 'level'])
    with pd.ExcelWriter(grouped_output_file, engine='openpyxl') as writer:
        level_group.to_excel(writer, sheet_name='level_group', index=False)
        city_group.to_excel(writer, sheet_name='city_group', index=False)
        category_group.to_excel(writer, sheet_name='city_type_group', index=False)
        category_level_group.to_excel(writer, sheet_name='city_level_group', index=False)

def main():
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset directory not found: {dataset_dir}')
    summary_rows = []
    for year in YEARS:
        for scenario in SCENARIOS:
            input_file = make_dataset_path(year, scenario)
            joint_output_file, grouped_output_file = make_outputs(year, scenario)
            if not input_file.exists():
                print(f'Skipping missing input: {input_file}')
                summary_rows.append({'year': year, 'scenario': scenario, 'status': 'missing_input', 'input_file': str(input_file), 'joint_output_file': str(joint_output_file), 'grouped_output_file': str(grouped_output_file)})
                continue
            if SKIP_EXISTING and joint_output_file.exists() and grouped_output_file.exists():
                print(f'Skipping existing results: {year}-{scenario}')
                summary_rows.append({'year': year, 'scenario': scenario, 'status': 'skipped_existing', 'input_file': str(input_file), 'joint_output_file': str(joint_output_file), 'grouped_output_file': str(grouped_output_file)})
                continue
            print('=' * 80)
            print(f'Starting: {year}-{scenario}')
            print(f'Input: {input_file}')
            random_seed = calc_seed(year, scenario)
            run_one_dataset(input_file=input_file, joint_output_file=joint_output_file, grouped_output_file=grouped_output_file, random_seed=random_seed)
            print(f'Completed: {year}-{scenario}')
            print(f'Monte Carlo results: {joint_output_file}')
            print(f'Grouped mean results: {grouped_output_file}')
            summary_rows.append({'year': year, 'scenario': scenario, 'status': 'completed', 'input_file': str(input_file), 'joint_output_file': str(joint_output_file), 'grouped_output_file': str(grouped_output_file), 'random_seed': random_seed, 'N': N})
    summary_file = output_dir / 'future_waste_env_MC_batch_summary.xlsx'
    pd.DataFrame(summary_rows).to_excel(summary_file, index=False, engine='openpyxl')
    print('=' * 80)
    print(f'All batch tasks completed; summary: {summary_file}')
if __name__ == '__main__':
    main()
