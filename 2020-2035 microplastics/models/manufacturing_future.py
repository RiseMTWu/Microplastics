import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')
INPUT_DIR = BASE_DIR / 'Manufacturing-dataset'
OUTPUT_DIR = BASE_DIR / 'results' / 'Manufacturing'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = range(2020, 2036)
INPUT_SHEET = 'prod-manu'
N, MIN_VALUE, RANDOM_SEED = (int(os.environ.get('MC_DRAWS', '10000')), 0.0001, 42)
PRO_COL, MANU_COL = ('pro_cbe/kt', 'manu_cbe/kt')
sources = ['Production', 'Manufacture']
fates = ['soil', 'ocean', 'river']
stat_names = ['mean', 'std', 'q05', 'q25', 'q75', 'q95']
indicator_columns = [f'{s}_MP' for s in sources] + ['total_MP'] + [f'{s}_{f}' for s in sources for f in fates] + [f'{s}_environment' for s in sources] + [f'total_{f}' for f in fates] + ['total_environment']
result_columns = [f'{indicator}_{stat_name}' for indicator in indicator_columns for stat_name in stat_names]
parameter_columns = ['AG11', 'AG12', 'TC16', 'TC13', 'TC14', 'TC15', 'AG21', 'AG22', 'TC26', 'TC23', 'TC24', 'TC25', 'TC315', 'TC414', 'TC57', 'TC58', 'TC59', 'TC510', 'TC614', 'TC615', 'TC714', 'TC814', 'TC811', 'TC914', 'TC911', 'TC1014', 'TC1011', 'TC1112', 'TC1113', 'TC1215', 'TC1417', 'TC1418', 'TC1516', 'TC1517', 'TC1718', 'TC1719']
required_columns = ['city', 'Category', 'level', 'province', PRO_COL, MANU_COL] + parameter_columns

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

def statistics(values):
    return [np.mean(values), np.std(values), *np.quantile(values, [0.05, 0.25, 0.75, 0.95])]

def empty_source_dict():
    return {source: np.zeros(N) for source in sources}

def simulate_one_row(row):
    pos = lambda name, cv: normal_positive(row[name], cv)
    frac = lambda name, cv: normal_fraction(row[name], cv)
    A1, A2 = (pos(PRO_COL, 0.16), pos(MANU_COL, 0.16))
    initial_MP = {'Production': A1, 'Manufacture': A2}
    A3, A4, A5, A6 = (empty_source_dict(), empty_source_dict(), empty_source_dict(), empty_source_dict())
    A14, A15, A16, A17, A18, A19 = [empty_source_dict() for _ in range(6)]
    RAG11, RAG12, RTC16 = normalize_arrays([pos('AG11', 0.43), pos('AG12', 0.43), pos('TC16', 0.43)])
    G1_prod, G2_prod = (A1 * RAG11, A1 * RAG12)
    A6['Production'] += A1 * RTC16
    RTC13, RTC14, RTC15 = normalize_arrays([pos('TC13', 0.26), pos('TC14', 0.25), pos('TC15', 0.43)])
    A3['Production'] += G1_prod * RTC13
    A4['Production'] += G1_prod * RTC14
    A5['Production'] += G1_prod * RTC15
    RTC23, RTC24, RTC25 = normalize_arrays([pos('TC23', 0.26), pos('TC24', 0.25), pos('TC25', 0.43)])
    A3['Production'] += G2_prod * RTC23
    A4['Production'] += G2_prod * RTC24
    A5['Production'] += G2_prod * RTC25
    RAG21, RAG22, RTC26 = normalize_arrays([pos('AG21', 0.43), pos('AG22', 0.43), pos('TC26', 0.43)])
    G1_manu, G2_manu = (A2 * RAG21, A2 * RAG22)
    A6['Manufacture'] += A2 * RTC26
    A3['Manufacture'] += G1_manu * RTC13 + G2_manu * RTC23
    A4['Manufacture'] += G1_manu * RTC14 + G2_manu * RTC24
    A5['Manufacture'] += G1_manu * RTC15 + G2_manu * RTC25
    TC315, TC414 = (frac('TC315', 0.05), frac('TC414', 0.05))
    RTC614, RTC615 = normalize_arrays([pos('TC614', 0.25), pos('TC615', 0.25)])
    for source in sources:
        A15[source] += A3[source] * TC315
        A14[source] += A4[source] * TC414 + A6[source] * RTC614
        A15[source] += A6[source] * RTC615
    RTC57, RTC58, RTC59, RTC510 = normalize_arrays([pos('TC57', 0.16), pos('TC58', 0.16), pos('TC59', 0.16), pos('TC510', 0.16)])
    TC714 = frac('TC714', 0.05)
    RTC814, RTC811 = normalize_arrays([pos('TC814', 0.16), pos('TC811', 0.16)])
    RTC914, RTC911 = normalize_arrays([pos('TC914', 0.16), pos('TC911', 0.16)])
    RTC1014, RTC1011 = normalize_arrays([pos('TC1014', 0.16), pos('TC1011', 0.16)])
    RTC1112, _ = normalize_arrays([pos('TC1112', 0.16), pos('TC1113', 0.16)])
    TC1215 = frac('TC1215', 0.05)
    for source in sources:
        A7, A8 = (A5[source] * RTC57, A5[source] * RTC58)
        A9, A10 = (A5[source] * RTC59, A5[source] * RTC510)
        A14[source] += A7 * TC714 + A8 * RTC814 + A9 * RTC914 + A10 * RTC1014
        A11 = A8 * RTC811 + A9 * RTC911 + A10 * RTC1011
        A15[source] += A11 * RTC1112 * TC1215
    RTC1417, RTC1418 = normalize_arrays([pos('TC1417', 0.42), pos('TC1418', 0.42)])
    RTC1516, RTC1517 = normalize_arrays([pos('TC1516', 0.46), pos('TC1517', 0.46)])
    RTC1718, RTC1719 = normalize_arrays([pos('TC1718', 0.16), pos('TC1719', 0.16)])
    for source in sources:
        A17[source] += A14[source] * RTC1417
        A18[source] += A14[source] * RTC1418
        A16[source] += A15[source] * RTC1516
        A17[source] += A15[source] * RTC1517
        A18[source] += A17[source] * RTC1718
        A19[source] += A17[source] * RTC1719
    source_environment = {source: A16[source] + A18[source] + A19[source] for source in sources}
    total_soil = sum((A16[source] for source in sources))
    total_ocean = sum((A18[source] for source in sources))
    total_river = sum((A19[source] for source in sources))
    total_environment = total_soil + total_ocean + total_river
    values = {'Production_MP': initial_MP['Production'], 'Manufacture_MP': initial_MP['Manufacture'], 'total_MP': A1 + A2, 'Production_soil': A16['Production'], 'Production_ocean': A18['Production'], 'Production_river': A19['Production'], 'Manufacture_soil': A16['Manufacture'], 'Manufacture_ocean': A18['Manufacture'], 'Manufacture_river': A19['Manufacture'], 'Production_environment': source_environment['Production'], 'Manufacture_environment': source_environment['Manufacture'], 'total_soil': total_soil, 'total_ocean': total_ocean, 'total_river': total_river, 'total_environment': total_environment}
    result = []
    for indicator in indicator_columns:
        result.extend(statistics(values[indicator]))
    return result

def save_mc_result(df, result_df, output_file):
    result_df.insert(0, 'city', df['city'].values)
    result_df.insert(1, 'Category', df['Category'].values)
    result_df.insert(2, 'level', df['level'].values)
    result_df.insert(3, 'province', df['province'].values)
    base_info = result_df[['city', 'Category', 'level', 'province']]
    statistic_frames = {}
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for stat_name in stat_names:
            stat_cols = [column for column in result_df.columns if column.endswith('_' + stat_name)]
            stat_df = pd.concat([base_info, result_df[stat_cols]], axis=1)
            stat_df = stat_df.rename(columns={column: column.removesuffix('_' + stat_name) for column in stat_cols})
            stat_df.to_excel(writer, sheet_name=stat_name, index=False)
            statistic_frames[stat_name] = stat_df
    return statistic_frames

def run_one_year(year):
    input_file = INPUT_DIR / f'{year}.xlsx'
    mc_output = OUTPUT_DIR / f'{year}_prod_manu_city_pop_result_MC.xlsx'
    if not input_file.exists():
        raise FileNotFoundError(f'{year} input file not found: {input_file}')
    print(f"\n{'=' * 70}\nStarting {year}\nInput file: {input_file}\n{'=' * 70}")
    np.random.seed(RANDOM_SEED)
    df = pd.read_excel(input_file, sheet_name=INPUT_SHEET)
    missing_cols = [column for column in required_columns if column not in df.columns]
    if missing_cols:
        raise ValueError(f'{input_file.name} missing required fields:\n' + ', '.join(missing_cols))
    records = df.to_dict(orient='records')
    result_df = pd.DataFrame(np.nan, index=range(len(df)), columns=result_columns)
    for index, row in enumerate(records):
        result_df.iloc[index] = simulate_one_row(row)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f'{year} progress: {index + 1}/{len(records)}')
    statistic_frames = save_mc_result(df, result_df, mc_output)
    print(f'{year} completed')
    print(f'Monte Carlo results: {mc_output}')

def main():
    for year in YEARS:
        run_one_year(year)
    print(f'\nAll years completed.\nResults directory: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()
