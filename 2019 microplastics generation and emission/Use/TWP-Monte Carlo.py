# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:45:19 2026

@author: w1077
"""
######################################### TWP
import pandas as pd
import warnings
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

N = 10000
MIN_VALUE = 0.0001
np.random.seed(42)

stat_names = ["mean", "std", "q05", "q25", "q75", "q95"]


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
    """Return the mean, standard deviation, and uncertainty quantiles."""
    out = [np.mean(x), np.std(x)]
    out.extend(np.quantile(x, [0.05, 0.25, 0.75, 0.95]))
    return out


def build_result_table(data_input, indicator_columns):
    result_columns = []

    for indicator in indicator_columns:
        for stat_name in stat_names:
            result_columns.append(f"{indicator}_{stat_name}")

    return pd.DataFrame(
        np.nan,
        index=range(len(data_input)),
        columns=result_columns
    )


def add_base_info(result_df, data_input):
    result_df.insert(0, "city", data_input["City"])
    result_df.insert(1, "Category", data_input["Category"] if "Category" in data_input.columns else np.nan)
    result_df.insert(2, "level", data_input["level"] if "level" in data_input.columns else np.nan)
    result_df.insert(3, "province", data_input["Province"] if "Province" in data_input.columns else np.nan)

    return result_df


def write_stat_sheets(result_df, output_file):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    base_info = result_df[["city", "Category", "level", "province"]]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for stat_name in stat_names:
            stat_cols = [
                col for col in result_df.columns
                if col.endswith("_" + stat_name)
            ]

            sheet_df = pd.concat([base_info, result_df[stat_cols]], axis=1)
            sheet_df = sheet_df.rename(
                columns={col: col.replace("_" + stat_name, "") for col in stat_cols}
            )

            sheet_df.to_excel(writer, sheet_name=stat_name, index=False)


input_file = BASE_DIR / "2019_TWP_dataset.xlsx"
output_file = BASE_DIR / "results" / "TWP_result_MC.xlsx"

data_input = pd.read_excel(input_file, sheet_name="2019_TWP_dataset")
print(data_input.shape)

indicator_columns = [
    "TWP",
    "vehicle_TWP",
    "airplane_TWP",
    "TWP_soil",
    "TWP_ocean",
    "TWP_river",
    "TWP_environment"
]

city_result = build_result_table(data_input, indicator_columns)

micro_columns = [
    "total", "BOX1", "BOX2", "BOX3", "BOX4", "BOX5", "BOX6", "BOX7", "BOX8", "BOX9", "BOX10",
    "BOX11", "BOX12", "BOX13", "BOX14", "BOX15", "BOX16", "BOX17", "BOX18"
]

micro = pd.DataFrame(np.nan, columns=micro_columns, index=range(N))


for i in range(len(data_input)):

    # =========================
    # 1. Vehicle activity and tire-wear factors
    # =========================
    small1 = normal_positive(data_input.loc[i, "small1"], 0.25)
    urban_A = normal_positive(data_input.loc[i, "cityroadA"], 0.28)
    M1 = normal_positive(data_input.loc[i, "M1"], 0.26)
    rural_A = normal_positive(data_input.loc[i, "ruralroadA"], 0.28)
    M2 = normal_positive(data_input.loc[i, "M2"], 0.26)
    highway_A = normal_positive(data_input.loc[i, "highwayA"], 0.28)
    M3 = normal_positive(data_input.loc[i, "M3"], 0.26)

    medium1 = normal_positive(data_input.loc[i, "medium1"], 0.25)
    urban_B = normal_positive(data_input.loc[i, "cityroadB"], 0.28)
    M4 = normal_positive(data_input.loc[i, "M4"], 0.26)
    rural_B = normal_positive(data_input.loc[i, "ruralroadB"], 0.28)
    M5 = normal_positive(data_input.loc[i, "M5"], 0.26)
    highway_B = normal_positive(data_input.loc[i, "highwayB"], 0.28)
    M6 = normal_positive(data_input.loc[i, "M6"], 0.26)

    big1 = normal_positive(data_input.loc[i, "big1"], 0.25)
    urban_C = normal_positive(data_input.loc[i, "cityroadC"], 0.28)
    M7 = normal_positive(data_input.loc[i, "M7"], 0.26)
    rural_C = normal_positive(data_input.loc[i, "ruralroadC"], 0.28)
    M8 = normal_positive(data_input.loc[i, "M8"], 0.26)
    highway_C = normal_positive(data_input.loc[i, "highwayC"], 0.28)
    M9 = normal_positive(data_input.loc[i, "M9"], 0.26)

    small2 = normal_positive(data_input.loc[i, "small2"], 0.25)
    urban_D = normal_positive(data_input.loc[i, "cityroadD"], 0.28)
    M10 = normal_positive(data_input.loc[i, "M10"], 0.26)
    rural_D = normal_positive(data_input.loc[i, "ruralroadD"], 0.28)
    M11 = normal_positive(data_input.loc[i, "M11"], 0.26)
    highway_D = normal_positive(data_input.loc[i, "highwayD"], 0.28)
    M12 = normal_positive(data_input.loc[i, "M12"], 0.26)

    medium2 = normal_positive(data_input.loc[i, "medium2"], 0.25)
    urban_E = normal_positive(data_input.loc[i, "cityroadE"], 0.28)
    M13 = normal_positive(data_input.loc[i, "M13"], 0.26)
    rural_E = normal_positive(data_input.loc[i, "ruralroadE"], 0.28)
    M14 = normal_positive(data_input.loc[i, "M14"], 0.26)
    highway_E = normal_positive(data_input.loc[i, "highwayE"], 0.28)
    M15 = normal_positive(data_input.loc[i, "M15"], 0.26)

    big2 = normal_positive(data_input.loc[i, "big2"], 0.25)
    urban_F = normal_positive(data_input.loc[i, "cityroadF"], 0.28)
    M16 = normal_positive(data_input.loc[i, "M16"], 0.26)
    rural_F = normal_positive(data_input.loc[i, "ruralroadF"], 0.28)
    M17 = normal_positive(data_input.loc[i, "M17"], 0.26)
    highway_F = normal_positive(data_input.loc[i, "highwayF"], 0.28)
    M18 = normal_positive(data_input.loc[i, "M18"], 0.26)

    # Clip fractional parameters to [0, 1]
    passenger_ratio = normal_fraction(data_input.loc[i, "Car"], 0.28)
    truck_ratio = normal_fraction(data_input.loc[i, "Truck"], 0.28)
    tire_synthetic_ratio = normal_fraction(data_input.loc[i, "TCR"], 0.11)

    airplane_input = normal_positive(data_input.loc[i, "Airplaneinput"], 0.28)

    # =========================
    # 2. Transfer coefficients
    # =========================
    TC13 = normal_positive(data_input.loc[i, "TC13"], 0.16)
    TC14 = normal_positive(data_input.loc[i, "TC14"], 0.16)
    TC15 = normal_positive(data_input.loc[i, "TC15"], 0.16)
    RTC13, RTC14, RTC15 = normalize_arrays([TC13, TC14, TC15])

    TC23 = normal_positive(data_input.loc[i, "TC23"], 0.16)
    TC24 = normal_positive(data_input.loc[i, "TC24"], 0.16)
    TC25 = normal_positive(data_input.loc[i, "TC25"], 0.16)
    RTC23, RTC24, RTC25 = normalize_arrays([TC23, TC24, TC25])

    # Clip the effective single-pathway transfer fraction to [0, 1]
    RTC314 = normal_fraction(data_input.loc[i, "TC314"], 0.05)
    RTC413 = normal_fraction(data_input.loc[i, "TC413"], 0.05)

    TC56 = normal_positive(data_input.loc[i, "TC56"], 0.16)
    TC57 = normal_positive(data_input.loc[i, "TC57"], 0.16)
    TC58 = normal_positive(data_input.loc[i, "TC58"], 0.16)
    TC59 = normal_positive(data_input.loc[i, "TC59"], 0.16)
    RTC56, RTC57, RTC58, RTC59 = normalize_arrays([TC56, TC57, TC58, TC59])

    RTC613 = normal_fraction(data_input.loc[i, "TC613"], 0.05)

    TC710 = normal_positive(data_input.loc[i, "TC710"], 0.16)
    TC713 = normal_positive(data_input.loc[i, "TC713"], 0.16)
    RTC710, RTC713 = normalize_arrays([TC710, TC713])

    TC810 = normal_positive(data_input.loc[i, "TC810"], 0.16)
    TC813 = normal_positive(data_input.loc[i, "TC813"], 0.16)
    RTC810, RTC813 = normalize_arrays([TC810, TC813])

    TC910 = normal_positive(data_input.loc[i, "TC910"], 0.16)
    TC913 = normal_positive(data_input.loc[i, "TC913"], 0.16)
    RTC910, RTC913 = normalize_arrays([TC910, TC913])

    TC1011 = normal_positive(data_input.loc[i, "TC1011"], 0.16)
    TC1012 = normal_positive(data_input.loc[i, "TC1012"], 0.16)
    RTC1011, RTC1012 = normalize_arrays([TC1011, TC1012])

    RTC1114 = normal_fraction(data_input.loc[i, "TC1114"], 0.05)

    TC1315 = normal_positive(data_input.loc[i, "TC1315"], 0.42)
    TC1316 = normal_positive(data_input.loc[i, "TC1316"], 0.42)
    RTC1315, RTC1316 = normalize_arrays([TC1315, TC1316])

    TC1415 = normal_positive(data_input.loc[i, "TC1415"], 0.46)
    TC1417 = normal_positive(data_input.loc[i, "TC1417"], 0.46)
    RTC1415, RTC1417 = normalize_arrays([TC1415, TC1417])

    TC1516 = normal_positive(data_input.loc[i, "TC1516"], 0.16)
    TC1518 = normal_positive(data_input.loc[i, "TC1518"], 0.16)
    RTC1516, RTC1518 = normalize_arrays([TC1516, TC1518])

    # =========================
    # 3. Tire-wear particle generation
    # =========================
    passenger_twp = (
        small1 * (urban_A * M1 + rural_A * M2 + highway_A * M3)
        + medium1 * (urban_B * M4 + rural_B * M5 + highway_B * M6)
        + big1 * (urban_C * M7 + rural_C * M8 + highway_C * M9)
    ) * tire_synthetic_ratio * passenger_ratio * 0.00001

    truck_twp = (
        small2 * (urban_D * M10 + rural_D * M11 + highway_D * M12)
        + medium2 * (urban_E * M13 + rural_E * M14 + highway_E * M15)
        + big2 * (urban_F * M16 + rural_F * M17 + highway_F * M18)
    ) * tire_synthetic_ratio * truck_ratio * 0.00001

    micro["BOX1"] = passenger_twp + truck_twp
    micro["BOX2"] = airplane_input
    micro["total"] = micro["BOX1"] + micro["BOX2"]

    # =========================
    # 4. Environmental fate
    # =========================
    micro["BOX3"] = RTC13 * micro["BOX1"] + RTC23 * micro["BOX2"]
    micro["BOX4"] = RTC14 * micro["BOX1"] + RTC24 * micro["BOX2"]
    micro["BOX5"] = RTC15 * micro["BOX1"] + RTC25 * micro["BOX2"]

    micro["BOX6"] = micro["BOX5"] * RTC56
    micro["BOX7"] = micro["BOX5"] * RTC57
    micro["BOX8"] = micro["BOX5"] * RTC58
    micro["BOX9"] = micro["BOX5"] * RTC59

    micro["BOX10"] = (
        micro["BOX7"] * RTC710
        + micro["BOX8"] * RTC810
        + micro["BOX9"] * RTC910
    )

    micro["BOX11"] = micro["BOX10"] * RTC1011
    micro["BOX12"] = micro["BOX10"] * RTC1012

    micro["BOX13"] = (
        micro["BOX7"] * RTC713
        + micro["BOX8"] * RTC813
        + micro["BOX9"] * RTC913
        + micro["BOX6"] * RTC613
        + micro["BOX4"] * RTC413
    )

    micro["BOX14"] = micro["BOX11"] * RTC1114 + micro["BOX3"] * RTC314

    micro["BOX15"] = micro["BOX13"] * RTC1315 + micro["BOX14"] * RTC1415
    micro["BOX16"] = micro["BOX13"] * RTC1316 + micro["BOX15"] * RTC1516
    micro["BOX17"] = micro["BOX14"] * RTC1417
    micro["BOX18"] = micro["BOX15"] * RTC1518

    values_dict = {
        "TWP": micro["total"],
        "vehicle_TWP": micro["BOX1"],
        "airplane_TWP": micro["BOX2"],
        "TWP_soil": micro["BOX17"],
        "TWP_ocean": micro["BOX16"],
        "TWP_river": micro["BOX18"],
        "TWP_environment": micro["BOX16"] + micro["BOX17"] + micro["BOX18"]
    }

    row_result = []

    for indicator in indicator_columns:
        row_result.extend(stat(values_dict[indicator]))

    city_result.iloc[i] = row_result


city_result = add_base_info(city_result, data_input)
write_stat_sheets(city_result, output_file)

print("TWP Monte Carlo simulation completed.")
print(f"Results saved to: {output_file}")