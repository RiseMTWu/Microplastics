# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:50:50 2026

@author: w1077
"""
#####################   Microplastic generation and environmental fate: city dataset

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

# =========================
# 1. File paths and basic parameters
# =========================
input_file = BASE_DIR / "pro-manu-env-city.xlsx"
output_file = BASE_DIR / "results" / "prod_manu_city_result_MC.xlsx"

Path(output_file).parent.mkdir(parents=True, exist_ok=True)

N = 10000
MIN_VALUE = 0.0001

# Fix the random seed for reproducibility
np.random.seed(42)

df = pd.read_excel(input_file, sheet_name="prod-manu")
n = len(df)

sources = ["Production", "Manufacture"]
fates = ["soil", "ocean", "river"]
stat_names = ["mean", "std", "q05", "q25", "q75", "q95"]

pro_col = "pro_pbe/kt"
manu_col = "manu_pbe/kt"


# =========================
# 2. Build output indicator columns
# =========================
indicator_columns = []

for s in sources:
    indicator_columns.append(f"{s}_MP")

indicator_columns.append("total_MP")

for s in sources:
    for f in fates:
        indicator_columns.append(f"{s}_{f}")

# Calculate total environmental emissions separately for Production and Manufacture
for s in sources:
    indicator_columns.append(f"{s}_environment")

for f in fates:
    indicator_columns.append(f"total_{f}")

indicator_columns.append("total_environment")

result_columns = []

for indicator in indicator_columns:
    for stat_name in stat_names:
        result_columns.append(f"{indicator}_{stat_name}")

city_result = pd.DataFrame(
    np.nan,
    index=range(n),
    columns=result_columns
)


# =========================
# 3. Monte Carlo helper functions
# =========================
def normal_positive(mean, cv, size=N):
    """Sample normally distributed parameters and replace negative draws with a small positive value."""
    mean = pd.to_numeric(mean, errors="coerce")
    mean = 0 if pd.isna(mean) else float(mean)

    x = np.random.normal(loc=mean, scale=abs(cv * mean), size=size)
    x[x < 0] = MIN_VALUE
    return x


def normal_fraction(mean, cv, size=N):
    """Sample fractional parameters and clip draws to the interval [0, 1]."""
    x = normal_positive(mean, cv, size)
    return np.clip(x, 0, 1)


def normalize_arrays(arrays):
    """Normalize outgoing fractions at each node so that they sum to one."""
    total = np.zeros(N)

    for a in arrays:
        total += a

    total[total <= 0] = MIN_VALUE

    return [a / total for a in arrays]


def stat(x):
    """Calculate the mean, standard deviation, and quantiles of Monte Carlo results."""
    out = [np.mean(x), np.std(x)]
    out.extend(np.quantile(x, [0.05, 0.25, 0.75, 0.95]))
    return out


# =========================
# 4. Calculate results for each city
# =========================
for i in range(n):

    # Initial microplastic generation, in kt
    A1_direct = normal_positive(df.loc[i, pro_col], 0.16)
    A2_direct = normal_positive(df.loc[i, manu_col], 0.16)

    initial_MP = {
        "Production": A1_direct,
        "Manufacture": A2_direct,
    }

    # Initialize intermediate nodes and final environmental compartments
    A3_soil_deposition = {s: np.zeros(N) for s in sources}
    A4_runoff_water = {s: np.zeros(N) for s in sources}
    A5_wastewater = {s: np.zeros(N) for s in sources}
    A6_airborne = {s: np.zeros(N) for s in sources}

    A14_aquatic = {s: np.zeros(N) for s in sources}
    A15_soil = {s: np.zeros(N) for s in sources}
    A16_residual_soil = {s: np.zeros(N) for s in sources}
    A17_river = {s: np.zeros(N) for s in sources}
    A18_ocean = {s: np.zeros(N) for s in sources}
    A19_residual_river = {s: np.zeros(N) for s in sources}

    # =========================
    # 4.1 Internal allocation within Production
    # =========================
    AG11 = normal_positive(df.loc[i, "AG11"], 0.43)
    AG12 = normal_positive(df.loc[i, "AG12"], 0.43)
    TC16 = normal_positive(df.loc[i, "TC16"], 0.43)

    RAG11, RAG12, RTC16 = normalize_arrays([AG11, AG12, TC16])

    G1_solid_prod = A1_direct * RAG11
    G2_washing_prod = A1_direct * RAG12
    A6_airborne["Production"] += A1_direct * RTC16

    TC13 = normal_positive(df.loc[i, "TC13"], 0.26)
    TC14 = normal_positive(df.loc[i, "TC14"], 0.25)
    TC15 = normal_positive(df.loc[i, "TC15"], 0.43)

    RTC13, RTC14, RTC15 = normalize_arrays([TC13, TC14, TC15])

    A3_soil_deposition["Production"] += G1_solid_prod * RTC13
    A4_runoff_water["Production"] += G1_solid_prod * RTC14
    A5_wastewater["Production"] += G1_solid_prod * RTC15

    TC23 = normal_positive(df.loc[i, "TC23"], 0.26)
    TC24 = normal_positive(df.loc[i, "TC24"], 0.25)
    TC25 = normal_positive(df.loc[i, "TC25"], 0.43)

    RTC23, RTC24, RTC25 = normalize_arrays([TC23, TC24, TC25])

    A3_soil_deposition["Production"] += G2_washing_prod * RTC23
    A4_runoff_water["Production"] += G2_washing_prod * RTC24
    A5_wastewater["Production"] += G2_washing_prod * RTC25

    # =========================
    # 4.2 Internal allocation within Manufacture
    # =========================
    AG21 = normal_positive(df.loc[i, "AG21"], 0.43)
    AG22 = normal_positive(df.loc[i, "AG22"], 0.43)
    TC26 = normal_positive(df.loc[i, "TC26"], 0.43)

    RAG21, RAG22, RTC26 = normalize_arrays([AG21, AG22, TC26])

    G1_solid_manu = A2_direct * RAG21
    G2_washing_manu = A2_direct * RAG22
    A6_airborne["Manufacture"] += A2_direct * RTC26

    A3_soil_deposition["Manufacture"] += G1_solid_manu * RTC13
    A4_runoff_water["Manufacture"] += G1_solid_manu * RTC14
    A5_wastewater["Manufacture"] += G1_solid_manu * RTC15

    A3_soil_deposition["Manufacture"] += G2_washing_manu * RTC23
    A4_runoff_water["Manufacture"] += G2_washing_manu * RTC24
    A5_wastewater["Manufacture"] += G2_washing_manu * RTC25

    # =========================
    # 4.3 Direct environmental release pathways
    # =========================
    TC315 = normal_fraction(df.loc[i, "TC315"], 0.05)
    TC414 = normal_fraction(df.loc[i, "TC414"], 0.05)

    TC614 = normal_positive(df.loc[i, "TC614"], 0.25)
    TC615 = normal_positive(df.loc[i, "TC615"], 0.25)

    RTC614, RTC615 = normalize_arrays([TC614, TC615])

    for s in sources:
        A15_soil[s] += A3_soil_deposition[s] * TC315
        A14_aquatic[s] += A4_runoff_water[s] * TC414

        A14_aquatic[s] += A6_airborne[s] * RTC614
        A15_soil[s] += A6_airborne[s] * RTC615

    # =========================
    # 4.4 Wastewater system pathways
    # =========================
    TC57 = normal_positive(df.loc[i, "TC57"], 0.16)
    TC58 = normal_positive(df.loc[i, "TC58"], 0.16)
    TC59 = normal_positive(df.loc[i, "TC59"], 0.16)
    TC510 = normal_positive(df.loc[i, "TC510"], 0.16)

    RTC57, RTC58, RTC59, RTC510 = normalize_arrays([TC57, TC58, TC59, TC510])

    TC714 = normal_fraction(df.loc[i, "TC714"], 0.05)

    TC814 = normal_positive(df.loc[i, "TC814"], 0.16)
    TC811 = normal_positive(df.loc[i, "TC811"], 0.16)

    RTC814, RTC811 = normalize_arrays([TC814, TC811])

    TC914 = normal_positive(df.loc[i, "TC914"], 0.16)
    TC911 = normal_positive(df.loc[i, "TC911"], 0.16)

    RTC914, RTC911 = normalize_arrays([TC914, TC911])

    TC1014 = normal_positive(df.loc[i, "TC1014"], 0.16)
    TC1011 = normal_positive(df.loc[i, "TC1011"], 0.16)

    RTC1014, RTC1011 = normalize_arrays([TC1014, TC1011])

    TC1112 = normal_positive(df.loc[i, "TC1112"], 0.16)
    TC1113 = normal_positive(df.loc[i, "TC1113"], 0.16)

    RTC1112, RTC1113 = normalize_arrays([TC1112, TC1113])

    TC1215 = normal_fraction(df.loc[i, "TC1215"], 0.05)

    for s in sources:
        A7_storm_overflow = A5_wastewater[s] * RTC57
        A8_primary = A5_wastewater[s] * RTC58
        A9_secondary = A5_wastewater[s] * RTC59
        A10_tertiary = A5_wastewater[s] * RTC510

        A14_aquatic[s] += A7_storm_overflow * TC714

        A14_aquatic[s] += A8_primary * RTC814
        A11_from_A8 = A8_primary * RTC811

        A14_aquatic[s] += A9_secondary * RTC914
        A11_from_A9 = A9_secondary * RTC911

        A14_aquatic[s] += A10_tertiary * RTC1014
        A11_from_A10 = A10_tertiary * RTC1011

        A11_removal = A11_from_A8 + A11_from_A9 + A11_from_A10

        A12_sludge_reuse = A11_removal * RTC1112
        A15_soil[s] += A12_sludge_reuse * TC1215

    # =========================
    # 4.5 Further transport in aquatic and soil environments
    # =========================
    TC1417 = normal_positive(df.loc[i, "TC1417"], 0.42)
    TC1418 = normal_positive(df.loc[i, "TC1418"], 0.42)

    RTC1417, RTC1418 = normalize_arrays([TC1417, TC1418])

    TC1516 = normal_positive(df.loc[i, "TC1516"], 0.46)
    TC1517 = normal_positive(df.loc[i, "TC1517"], 0.46)

    RTC1516, RTC1517 = normalize_arrays([TC1516, TC1517])

    TC1718 = normal_positive(df.loc[i, "TC1718"], 0.16)
    TC1719 = normal_positive(df.loc[i, "TC1719"], 0.16)

    RTC1718, RTC1719 = normalize_arrays([TC1718, TC1719])

    for s in sources:
        A17_river[s] += A14_aquatic[s] * RTC1417
        A18_ocean[s] += A14_aquatic[s] * RTC1418

        A16_residual_soil[s] += A15_soil[s] * RTC1516
        A17_river[s] += A15_soil[s] * RTC1517

        A18_ocean[s] += A17_river[s] * RTC1718
        A19_residual_river[s] += A17_river[s] * RTC1719

    # =========================
    # 4.6 Summarize final environmental fate
    # =========================
    source_environment = {}

    for s in sources:
        source_environment[s] = (
            A16_residual_soil[s]
            + A18_ocean[s]
            + A19_residual_river[s]
        )

    total_A16_residual_soil = sum(A16_residual_soil[s] for s in sources)
    total_A18_ocean = sum(A18_ocean[s] for s in sources)
    total_A19_residual_river = sum(A19_residual_river[s] for s in sources)

    total_environment = (
        total_A16_residual_soil
        + total_A18_ocean
        + total_A19_residual_river
    )

    values_dict = {}

    for s in sources:
        values_dict[f"{s}_MP"] = initial_MP[s]

    values_dict["total_MP"] = A1_direct + A2_direct

    for s in sources:
        values_dict[f"{s}_soil"] = A16_residual_soil[s]
        values_dict[f"{s}_ocean"] = A18_ocean[s]
        values_dict[f"{s}_river"] = A19_residual_river[s]
        values_dict[f"{s}_environment"] = source_environment[s]

    values_dict["total_soil"] = total_A16_residual_soil
    values_dict["total_ocean"] = total_A18_ocean
    values_dict["total_river"] = total_A19_residual_river
    values_dict["total_environment"] = total_environment

    row_result = []

    for indicator in indicator_columns:
        row_result.extend(stat(values_dict[indicator]))

    city_result.iloc[i] = row_result


# =========================
# 5. Export results
# =========================
city_result.insert(0, "city", df["city"])
city_result.insert(1, "Category", df["Category"])
city_result.insert(2, "province", df["province"])

base_info = city_result[["city", "Category", "province"]]

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for stat_name in stat_names:

        stat_cols = [
            col for col in city_result.columns
            if col.endswith("_" + stat_name)
        ]

        sheet_df = pd.concat(
            [
                base_info,
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

print("Production and manufacturing microplastic generation and environmental fate completed.")
print(f"Results saved to: {output_file}")
#%%
#####################   Microplastic generation and environmental fate: city-population dataset

import pandas as pd
import numpy as np
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")


input_file = BASE_DIR / "pro-manu-env-citypop.xlsx"
output_file = BASE_DIR / "results" / "prod_manu_city_pop_result_MC.xlsx"

Path(output_file).parent.mkdir(parents=True, exist_ok=True)

N = 10000
MIN_VALUE = 0.0001

# Fix the random seed for reproducibility
np.random.seed(42)

df = pd.read_excel(input_file, sheet_name="prod-manu")
n = len(df)

sources = ["Production", "Manufacture"]
fates = ["soil", "ocean", "river"]
stat_names = ["mean", "std", "q05", "q25", "q75", "q95"]

pro_col = "pro_cbe/kt"
manu_col = "manu_cbe/kt"


# =========================
# 2. Build output indicator columns
# =========================
indicator_columns = []

for s in sources:
    indicator_columns.append(f"{s}_MP")

indicator_columns.append("total_MP")

for s in sources:
    for f in fates:
        indicator_columns.append(f"{s}_{f}")

# Calculate total environmental emissions separately for Production and Manufacture
for s in sources:
    indicator_columns.append(f"{s}_environment")

for f in fates:
    indicator_columns.append(f"total_{f}")

indicator_columns.append("total_environment")

result_columns = []

for indicator in indicator_columns:
    for stat_name in stat_names:
        result_columns.append(f"{indicator}_{stat_name}")

city_result = pd.DataFrame(
    np.nan,
    index=range(n),
    columns=result_columns
)


# =========================
# 3. Monte Carlo helper functions
# =========================
def normal_positive(mean, cv, size=N):
    """Sample normally distributed parameters and replace negative draws with a small positive value."""
    mean = pd.to_numeric(mean, errors="coerce")
    mean = 0 if pd.isna(mean) else float(mean)

    x = np.random.normal(loc=mean, scale=abs(cv * mean), size=size)
    x[x < 0] = MIN_VALUE
    return x


def normal_fraction(mean, cv, size=N):
    """Sample fractional parameters and clip draws to the interval [0, 1]."""
    x = normal_positive(mean, cv, size)
    return np.clip(x, 0, 1)


def normalize_arrays(arrays):
    """Normalize outgoing fractions at each node so that they sum to one."""
    total = np.zeros(N)

    for a in arrays:
        total += a

    total[total <= 0] = MIN_VALUE

    return [a / total for a in arrays]


def stat(x):
    """Calculate the mean, standard deviation, and quantiles of Monte Carlo results."""
    out = [np.mean(x), np.std(x)]
    out.extend(np.quantile(x, [0.05, 0.25, 0.75, 0.95]))
    return out


# =========================
# 4. Calculate results for each city
# =========================
for i in range(n):

    # Initial microplastic generation, in kt
    A1_direct = normal_positive(df.loc[i, pro_col], 0.16)
    A2_direct = normal_positive(df.loc[i, manu_col], 0.16)

    initial_MP = {
        "Production": A1_direct,
        "Manufacture": A2_direct,
    }

    # Initialize intermediate nodes and final environmental compartments
    A3_soil_deposition = {s: np.zeros(N) for s in sources}
    A4_runoff_water = {s: np.zeros(N) for s in sources}
    A5_wastewater = {s: np.zeros(N) for s in sources}
    A6_airborne = {s: np.zeros(N) for s in sources}

    A14_aquatic = {s: np.zeros(N) for s in sources}
    A15_soil = {s: np.zeros(N) for s in sources}
    A16_residual_soil = {s: np.zeros(N) for s in sources}
    A17_river = {s: np.zeros(N) for s in sources}
    A18_ocean = {s: np.zeros(N) for s in sources}
    A19_residual_river = {s: np.zeros(N) for s in sources}

    # =========================
    # 4.1 Internal allocation within Production
    # =========================
    AG11 = normal_positive(df.loc[i, "AG11"], 0.43)
    AG12 = normal_positive(df.loc[i, "AG12"], 0.43)
    TC16 = normal_positive(df.loc[i, "TC16"], 0.43)

    RAG11, RAG12, RTC16 = normalize_arrays([AG11, AG12, TC16])

    G1_solid_prod = A1_direct * RAG11
    G2_washing_prod = A1_direct * RAG12
    A6_airborne["Production"] += A1_direct * RTC16

    TC13 = normal_positive(df.loc[i, "TC13"], 0.26)
    TC14 = normal_positive(df.loc[i, "TC14"], 0.25)
    TC15 = normal_positive(df.loc[i, "TC15"], 0.43)

    RTC13, RTC14, RTC15 = normalize_arrays([TC13, TC14, TC15])

    A3_soil_deposition["Production"] += G1_solid_prod * RTC13
    A4_runoff_water["Production"] += G1_solid_prod * RTC14
    A5_wastewater["Production"] += G1_solid_prod * RTC15

    TC23 = normal_positive(df.loc[i, "TC23"], 0.26)
    TC24 = normal_positive(df.loc[i, "TC24"], 0.25)
    TC25 = normal_positive(df.loc[i, "TC25"], 0.43)

    RTC23, RTC24, RTC25 = normalize_arrays([TC23, TC24, TC25])

    A3_soil_deposition["Production"] += G2_washing_prod * RTC23
    A4_runoff_water["Production"] += G2_washing_prod * RTC24
    A5_wastewater["Production"] += G2_washing_prod * RTC25

    # =========================
    # 4.2 Internal allocation within Manufacture
    # =========================
    AG21 = normal_positive(df.loc[i, "AG21"], 0.43)
    AG22 = normal_positive(df.loc[i, "AG22"], 0.43)
    TC26 = normal_positive(df.loc[i, "TC26"], 0.43)

    RAG21, RAG22, RTC26 = normalize_arrays([AG21, AG22, TC26])

    G1_solid_manu = A2_direct * RAG21
    G2_washing_manu = A2_direct * RAG22
    A6_airborne["Manufacture"] += A2_direct * RTC26

    A3_soil_deposition["Manufacture"] += G1_solid_manu * RTC13
    A4_runoff_water["Manufacture"] += G1_solid_manu * RTC14
    A5_wastewater["Manufacture"] += G1_solid_manu * RTC15

    A3_soil_deposition["Manufacture"] += G2_washing_manu * RTC23
    A4_runoff_water["Manufacture"] += G2_washing_manu * RTC24
    A5_wastewater["Manufacture"] += G2_washing_manu * RTC25

    # =========================
    # 4.3 Direct environmental release pathways
    # =========================
    TC315 = normal_fraction(df.loc[i, "TC315"], 0.05)
    TC414 = normal_fraction(df.loc[i, "TC414"], 0.05)

    TC614 = normal_positive(df.loc[i, "TC614"], 0.25)
    TC615 = normal_positive(df.loc[i, "TC615"], 0.25)

    RTC614, RTC615 = normalize_arrays([TC614, TC615])

    for s in sources:
        A15_soil[s] += A3_soil_deposition[s] * TC315
        A14_aquatic[s] += A4_runoff_water[s] * TC414

        A14_aquatic[s] += A6_airborne[s] * RTC614
        A15_soil[s] += A6_airborne[s] * RTC615

    # =========================
    # 4.4 Wastewater system pathways
    # =========================
    TC57 = normal_positive(df.loc[i, "TC57"], 0.16)
    TC58 = normal_positive(df.loc[i, "TC58"], 0.16)
    TC59 = normal_positive(df.loc[i, "TC59"], 0.16)
    TC510 = normal_positive(df.loc[i, "TC510"], 0.16)

    RTC57, RTC58, RTC59, RTC510 = normalize_arrays([TC57, TC58, TC59, TC510])

    TC714 = normal_fraction(df.loc[i, "TC714"], 0.05)

    TC814 = normal_positive(df.loc[i, "TC814"], 0.16)
    TC811 = normal_positive(df.loc[i, "TC811"], 0.16)

    RTC814, RTC811 = normalize_arrays([TC814, TC811])

    TC914 = normal_positive(df.loc[i, "TC914"], 0.16)
    TC911 = normal_positive(df.loc[i, "TC911"], 0.16)

    RTC914, RTC911 = normalize_arrays([TC914, TC911])

    TC1014 = normal_positive(df.loc[i, "TC1014"], 0.16)
    TC1011 = normal_positive(df.loc[i, "TC1011"], 0.16)

    RTC1014, RTC1011 = normalize_arrays([TC1014, TC1011])

    TC1112 = normal_positive(df.loc[i, "TC1112"], 0.16)
    TC1113 = normal_positive(df.loc[i, "TC1113"], 0.16)

    RTC1112, RTC1113 = normalize_arrays([TC1112, TC1113])

    TC1215 = normal_fraction(df.loc[i, "TC1215"], 0.05)

    for s in sources:
        A7_storm_overflow = A5_wastewater[s] * RTC57
        A8_primary = A5_wastewater[s] * RTC58
        A9_secondary = A5_wastewater[s] * RTC59
        A10_tertiary = A5_wastewater[s] * RTC510

        A14_aquatic[s] += A7_storm_overflow * TC714

        A14_aquatic[s] += A8_primary * RTC814
        A11_from_A8 = A8_primary * RTC811

        A14_aquatic[s] += A9_secondary * RTC914
        A11_from_A9 = A9_secondary * RTC911

        A14_aquatic[s] += A10_tertiary * RTC1014
        A11_from_A10 = A10_tertiary * RTC1011

        A11_removal = A11_from_A8 + A11_from_A9 + A11_from_A10

        A12_sludge_reuse = A11_removal * RTC1112
        A15_soil[s] += A12_sludge_reuse * TC1215

    # =========================
    # 4.5 Further transport in aquatic and soil environments
    # =========================
    TC1417 = normal_positive(df.loc[i, "TC1417"], 0.42)
    TC1418 = normal_positive(df.loc[i, "TC1418"], 0.42)

    RTC1417, RTC1418 = normalize_arrays([TC1417, TC1418])

    TC1516 = normal_positive(df.loc[i, "TC1516"], 0.46)
    TC1517 = normal_positive(df.loc[i, "TC1517"], 0.46)

    RTC1516, RTC1517 = normalize_arrays([TC1516, TC1517])

    TC1718 = normal_positive(df.loc[i, "TC1718"], 0.16)
    TC1719 = normal_positive(df.loc[i, "TC1719"], 0.16)

    RTC1718, RTC1719 = normalize_arrays([TC1718, TC1719])

    for s in sources:
        A17_river[s] += A14_aquatic[s] * RTC1417
        A18_ocean[s] += A14_aquatic[s] * RTC1418

        A16_residual_soil[s] += A15_soil[s] * RTC1516
        A17_river[s] += A15_soil[s] * RTC1517

        A18_ocean[s] += A17_river[s] * RTC1718
        A19_residual_river[s] += A17_river[s] * RTC1719

    # =========================
    # 4.6 Summarize final environmental fate
    # =========================
    source_environment = {}

    for s in sources:
        source_environment[s] = (
            A16_residual_soil[s]
            + A18_ocean[s]
            + A19_residual_river[s]
        )

    total_A16_residual_soil = sum(A16_residual_soil[s] for s in sources)
    total_A18_ocean = sum(A18_ocean[s] for s in sources)
    total_A19_residual_river = sum(A19_residual_river[s] for s in sources)

    total_environment = (
        total_A16_residual_soil
        + total_A18_ocean
        + total_A19_residual_river
    )

    values_dict = {}

    for s in sources:
        values_dict[f"{s}_MP"] = initial_MP[s]

    values_dict["total_MP"] = A1_direct + A2_direct

    for s in sources:
        values_dict[f"{s}_soil"] = A16_residual_soil[s]
        values_dict[f"{s}_ocean"] = A18_ocean[s]
        values_dict[f"{s}_river"] = A19_residual_river[s]
        values_dict[f"{s}_environment"] = source_environment[s]

    values_dict["total_soil"] = total_A16_residual_soil
    values_dict["total_ocean"] = total_A18_ocean
    values_dict["total_river"] = total_A19_residual_river
    values_dict["total_environment"] = total_environment

    row_result = []

    for indicator in indicator_columns:
        row_result.extend(stat(values_dict[indicator]))

    city_result.iloc[i] = row_result


# =========================
# 5. Export results
# =========================
city_result.insert(0, "city", df["city"])
city_result.insert(1, "Category", df["Category"])
city_result.insert(2, "level", df["level"])
city_result.insert(3, "province", df["province"])

base_info = city_result[["city", "Category", "level","province"]]

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for stat_name in stat_names:

        stat_cols = [
            col for col in city_result.columns
            if col.endswith("_" + stat_name)
        ]

        sheet_df = pd.concat(
            [
                base_info,
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

print("Production and manufacturing microplastic generation and environmental fate completed.")
print(f"Results saved to: {output_file}")
