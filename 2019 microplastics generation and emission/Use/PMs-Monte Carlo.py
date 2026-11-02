# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:17:53 2026

@author: w1077
"""
##################### PMs Environmental fate：Monte Carlo
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

# =========================
# 1. Path settings
# =========================
input_file = BASE_DIR / "2019_PMs_dataset.xlsx"
output_file = BASE_DIR / "results" / "PMs_result_MC.xlsx"

output_file.parent.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Monte Carlo parameters
# =========================
N = 10000
MIN_VALUE = 0.0001
np.random.seed(42)

df = pd.read_excel(input_file)
n = len(df)

stat_names = ["mean", "std", "q05", "q25", "q75", "q95"]

result_indicators = [
    "PMs_toothpaste",
    "PMs_body_wash",
    "PMs_facial_scrub",
    "PMs_facial_cleanser",
    "Input",
    "D1_rinsed_off",
    "D13_residual_soil",
    "D15_ocean",
    "D16_residual_river",
    "total_environment",
]

result_columns = [
    f"{indicator}_{stat_name}"
    for indicator in result_indicators
    for stat_name in stat_names
]

city_result = pd.DataFrame(
    np.nan,
    index=range(n),
    columns=result_columns
)


# =========================
# 3. Sampling functions
# =========================
def normal_positive(mean, cv, size=N):
    """
    For non-fractional parameters:
    such as consumption, price per unit mass, and emission factors.
    Apply a non-negative lower bound without an upper bound.
    """
    mean = pd.to_numeric(mean, errors="coerce")
    mean = 0 if pd.isna(mean) else float(mean)

    x = np.random.normal(loc=mean, scale=abs(cv * mean), size=size)
    x[x < 0] = MIN_VALUE
    return x


def normal_fraction(mean, cv, size=N):
    """
    For fractional parameters:
    such as population shares, microplastic-containing product fractions, and single-pathway transfer fractions.
    Clip sampled values to [0, 1].
    """
    x = normal_positive(mean, cv, size)
    return np.clip(x, 0, 1)


def normalize_arrays(arrays):
    """
    For nodes with multiple outgoing pathways.
    Normalize sampled outgoing fractions so they sum to one.
    """
    total = np.zeros(N)

    for a in arrays:
        total += a

    total[total <= 0] = MIN_VALUE

    return [a / total for a in arrays]


def stat(x):
    """Return the mean, standard deviation, and quantiles."""
    out = [np.mean(x), np.std(x)]
    out.extend(np.quantile(x, [0.05, 0.25, 0.75, 0.95]))
    return out


# =========================
# 4. Main Monte Carlo loop
# =========================
for i in range(n):

    # -------------------------
    # 4.1 Sample basic parameters
    # -------------------------
    # Consumption: 100 million CNY; not a fraction
    Consumption_toothpaste = normal_positive(df.loc[i, "Consumption-toothpaste"], 0.15)
    Consumption_body_wash = normal_positive(df.loc[i, "Consumption-body_wash"], 0.15)
    Consumption_facial_scrub = normal_positive(df.loc[i, "Consumption-facial_scrub"], 0.15)
    Consumption_facial_cleanser = normal_positive(df.loc[i, "Consumption-facial_cleanser"], 0.15)

    # PCPpercent: population share in each of ten groups; a fraction
    PCPpercent = normal_fraction(df.loc[i, "PCPpercent"], 0.15)

    # PPUW: CNY/g; not a fraction
    PPUW_toothpaste = normal_positive(df.loc[i, "PPUW-toothpaste"], 0.15)
    PPUW_body_wash = normal_positive(df.loc[i, "PPUW-body_wash"], 0.15)
    PPUW_facial_scrub = normal_positive(df.loc[i, "PPUW-facial_scrub"], 0.15)
    PPUW_facial_cleanser = normal_positive(df.loc[i, "PPUW-facial_cleanser"], 0.15)

    # PMs-*: fraction of products containing microplastics
    PMs_toothpaste_percent = normal_fraction(df.loc[i, "PMs-toothpaste"], 0.13)
    PMs_body_wash_percent = normal_fraction(df.loc[i, "PMs-body_wash"], 0.13)
    PMs_facial_scrub_percent = normal_fraction(df.loc[i, "PMs-facial_scrub"], 0.13)
    PMs_facial_cleanser_percent = normal_fraction(df.loc[i, "PMs-facial_cleanser"], 0.13)

    # EF: mg/g; not a fraction
    EF_toothpaste = normal_positive(df.loc[i, "EF-toothpaste"], 0.13)
    EF_body_wash = normal_positive(df.loc[i, "EF-body_wash"], 0.13)
    EF_facial_scrub = normal_positive(df.loc[i, "EF-facial_scrub"], 0.13)
    EF_facial_cleanser = normal_positive(df.loc[i, "EF-facial_cleanser"], 0.13)

    # -------------------------
    # 4.2 Calculate PM generation from four personal-care product categories
    # Unit: tonnes
    # tonnes = Consumption(100 million CNY) * 1e8 * PCPpercent / PPUW(CNY/g)
    #      * PMs_fraction * EF(mg/g) / 1e9
    # The simplified conversion coefficient is 0.1
    # -------------------------
    PMs_toothpaste = (
        Consumption_toothpaste
        * PCPpercent
        / PPUW_toothpaste
        * PMs_toothpaste_percent
        * EF_toothpaste
        * 0.1
    )

    PMs_body_wash = (
        Consumption_body_wash
        * PCPpercent
        / PPUW_body_wash
        * PMs_body_wash_percent
        * EF_body_wash
        * 0.1
    )

    PMs_facial_scrub = (
        Consumption_facial_scrub
        * PCPpercent
        / PPUW_facial_scrub
        * PMs_facial_scrub_percent
        * EF_facial_scrub
        * 0.1
    )

    PMs_facial_cleanser = (
        Consumption_facial_cleanser
        * PCPpercent
        / PPUW_facial_cleanser
        * PMs_facial_cleanser_percent
        * EF_facial_cleanser
        * 0.1
    )

    Input = PMs_toothpaste + PMs_body_wash + PMs_facial_scrub + PMs_facial_cleanser

    # 95% is rinsed off into downstream pathways
    D1 = Input * 0.95

    # -------------------------
    # 4.3 D1: Rinsed off -> D2 runoff / D3 wastewater systems
    # Multiple outgoing pathways: normalize after sampling
    # -------------------------
    TC12 = normal_positive(df.loc[i, "TC12"], 0.43)
    TC13 = normal_positive(df.loc[i, "TC13"], 0.43)
    RTC12, RTC13 = normalize_arrays([TC12, TC13])

    D2 = D1 * RTC12
    D3 = D1 * RTC13

    # -------------------------
    # 4.4 D2 runoff -> D11 aquatic contamination
    # Effective single-pathway transfer fraction: clip to [0, 1]
    # -------------------------
    TC211 = normal_fraction(df.loc[i, "TC211"], 0.05)
    D11_from_D2 = D2 * TC211

    # -------------------------
    # 4.5 D3 wastewater systems -> D4/D5/D6/D7
    # Multiple outgoing pathways: normalize fractions
    # -------------------------
    TC34 = normal_positive(df.loc[i, "TC34"], 0.16)
    TC35 = normal_positive(df.loc[i, "TC35"], 0.16)
    TC36 = normal_positive(df.loc[i, "TC36"], 0.16)
    TC37 = normal_positive(df.loc[i, "TC37"], 0.16)
    RTC34, RTC35, RTC36, RTC37 = normalize_arrays([TC34, TC35, TC36, TC37])

    D4 = D3 * RTC34
    D5 = D3 * RTC35
    D6 = D3 * RTC36
    D7 = D3 * RTC37

    # -------------------------
    # 4.6 D4 stormwater overflow -> D11 aquatic contamination
    # Effective single-pathway transfer fraction: clip to [0, 1]
    # -------------------------
    TC411 = normal_fraction(df.loc[i, "TC411"], 0.05)
    D11_from_D4 = D4 * TC411

    # -------------------------
    # 4.7 D5 primary wastewater treatment -> D11 / D8
    # -------------------------
    TC511 = normal_positive(df.loc[i, "TC511"], 0.16)
    TC58 = normal_positive(df.loc[i, "TC58"], 0.16)
    RTC511, RTC58 = normalize_arrays([TC511, TC58])

    D11_from_D5 = D5 * RTC511
    D8_from_D5 = D5 * RTC58

    # -------------------------
    # 4.8 D6 secondary wastewater treatment -> D11 / D8
    # -------------------------
    TC611 = normal_positive(df.loc[i, "TC611"], 0.16)
    TC68 = normal_positive(df.loc[i, "TC68"], 0.16)
    RTC611, RTC68 = normalize_arrays([TC611, TC68])

    D11_from_D6 = D6 * RTC611
    D8_from_D6 = D6 * RTC68

    # -------------------------
    # 4.9 D7 tertiary wastewater treatment -> D11 / D8
    # -------------------------
    TC711 = normal_positive(df.loc[i, "TC711"], 0.16)
    TC78 = normal_positive(df.loc[i, "TC78"], 0.16)
    RTC711, RTC78 = normalize_arrays([TC711, TC78])

    D11_from_D7 = D7 * RTC711
    D8_from_D7 = D7 * RTC78

    # -------------------------
    # 4.10 D8 microplastic removal -> D9 sludge reuse / D10 proper disposal
    # -------------------------
    D8 = D8_from_D5 + D8_from_D6 + D8_from_D7

    TC89 = normal_positive(df.loc[i, "TC89"], 0.16)
    TC810 = normal_positive(df.loc[i, "TC810"], 0.16)
    RTC89, RTC810 = normalize_arrays([TC89, TC810])

    D9 = D8 * RTC89
    D10 = D8 * RTC810

    # -------------------------
    # 4.11 D9 sludge reuse -> D12 soil contamination
    # Effective single-pathway transfer fraction: clip to [0, 1]
    # -------------------------
    TC912 = normal_fraction(df.loc[i, "TC912"], 0.05)
    D12 = D9 * TC912

    # -------------------------
    # 4.12 D11 aquatic contamination -> D14 river / D15 ocean
    # -------------------------
    D11 = (
        D11_from_D2
        + D11_from_D4
        + D11_from_D5
        + D11_from_D6
        + D11_from_D7
    )

    TC1114 = normal_positive(df.loc[i, "TC1114"], 0.42)
    TC1115 = normal_positive(df.loc[i, "TC1115"], 0.42)
    RTC1114, RTC1115 = normalize_arrays([TC1114, TC1115])

    D14_from_D11 = D11 * RTC1114
    D15_from_D11 = D11 * RTC1115

    # -------------------------
    # 4.13 D12 soil contamination -> D13 residual soil / D14 river
    # -------------------------
    TC1213 = normal_positive(df.loc[i, "TC1213"], 0.46)
    TC1214 = normal_positive(df.loc[i, "TC1214"], 0.46)
    RTC1213, RTC1214 = normalize_arrays([TC1213, TC1214])

    D13 = D12 * RTC1213
    D14_from_D12 = D12 * RTC1214

    # -------------------------
    # 4.14 D14 river contamination -> D15 ocean / D16 residual river
    # -------------------------
    D14 = D14_from_D11 + D14_from_D12

    TC1415 = normal_positive(df.loc[i, "TC1415"], 0.16)
    TC1416 = normal_positive(df.loc[i, "TC1416"], 0.16)
    RTC1415, RTC1416 = normalize_arrays([TC1415, TC1416])

    D15 = D15_from_D11 + D14 * RTC1415
    D16 = D14 * RTC1416

    total_environment = D13 + D15 + D16

    # -------------------------
    # 4.15 Save Monte Carlo results for this row
    # -------------------------
    values_dict = {
        "PMs_toothpaste": PMs_toothpaste,
        "PMs_body_wash": PMs_body_wash,
        "PMs_facial_scrub": PMs_facial_scrub,
        "PMs_facial_cleanser": PMs_facial_cleanser,
        "Input": Input,
        "D1_rinsed_off": D1,
        "D13_residual_soil": D13,
        "D15_ocean": D15,
        "D16_residual_river": D16,
        "total_environment": total_environment,
    }

    row_result = []

    for indicator in result_indicators:
        row_result.extend(stat(values_dict[indicator]))

    city_result.iloc[i] = row_result


# =========================
# 5. Add identifying information
# =========================
base_cols = []

for col in ["city", "Category", "level", "province"]:
    if col in df.columns:
        city_result.insert(len(base_cols), col, df[col])
        base_cols.append(col)

# =========================
# 6. Save mean, standard deviation, and quantiles in separate worksheets
# =========================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for stat_name in stat_names:

        stat_cols = [
            col for col in city_result.columns
            if col.endswith("_" + stat_name)
        ]

        sheet_df = pd.concat(
            [
                city_result[base_cols],
                city_result[stat_cols]
            ],
            axis=1
        )

        rename_dict = {
            col: col.replace("_" + stat_name, "")
            for col in stat_cols
        }

        sheet_df = sheet_df.rename(columns=rename_dict)

        sheet_df.to_excel(
            writer,
            sheet_name=stat_name,
            index=False
        )

print("PMs environmental-fate Monte Carlo calculation completed.")
print(f"Results saved to: {output_file}")