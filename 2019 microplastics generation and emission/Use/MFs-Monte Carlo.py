# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:05:36 2026

@author: w1077
"""
################################# MFs

import pandas as pd
import numpy as np
import warnings
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


input_file = BASE_DIR / "2019_MFs_dataset.xlsx"
output_file = BASE_DIR / "results" / "MFs_result_MC.xlsx"

data_input = pd.read_excel(input_file, sheet_name="2019_MFs_dataset")
print(data_input.shape)

indicator_columns = [
    "MFs",
    "machine_MFs",
    "hand_MFs",
    "MFs_soil",
    "MFs_ocean",
    "MFs_river",
    "MFs_environment"
]

city_result = build_result_table(data_input, indicator_columns)

micro_columns = [
    "total", "BOX1", "BOX2", "BOX3", "BOX4", "BOX5", "BOX6", "BOX7", "BOX8", "BOX9", "BOX10",
    "BOX11", "BOX12", "BOX13", "BOX14", "BOX15", "BOX16", "BOX17", "BOX18"
]

micro = pd.DataFrame(np.nan, index=range(N), columns=micro_columns)


for i in range(len(data_input)):

    # =========================
    # 1. Parameters for microfiber generation
    # =========================
    population = normal_positive(data_input.loc[i, "population"], 0.42)

    # The synthetic-fiber proportion is fractional and is clipped to [0, 1]
    synthetic_fiber_ratio = normal_fraction(data_input.loc[i, "synthetic fibers"], 0.15)

    # Washing frequencies are not fractions and must not be clipped to [0, 1]
    machine_wash_freq = normal_positive(data_input.loc[i, "FrequencyMW"], 0.43)
    hand_wash_freq = normal_positive(data_input.loc[i, "FrequencyHW"], 0.43)

    machine_ef = normal_positive(data_input.loc[i, "EF-MW"], 0.17)
    hand_ef = normal_positive(data_input.loc[i, "EF-HW"], 0.17)

    machine_load = normal_positive(data_input.loc[i, "Load-MW"], 0.42)
    hand_load = normal_positive(data_input.loc[i, "Load-HW"], 0.42)

    # =========================
    # 2. Transfer coefficients
    # =========================
    TC14 = normal_positive(data_input.loc[i, "TC14"], 0.43)
    TC15 = normal_positive(data_input.loc[i, "TC15"], 0.43)
    RTC14, RTC15 = normalize_arrays([TC14, TC15])

    TC23 = normal_positive(data_input.loc[i, "TC23"], 0.43)
    TC24 = normal_positive(data_input.loc[i, "TC24"], 0.43)
    TC25 = normal_positive(data_input.loc[i, "TC25"], 0.43)
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
    # 3. Microfiber generation
    # =========================
    micro["BOX1"] = (
        population
        * synthetic_fiber_ratio
        * machine_wash_freq
        * machine_load
        * machine_ef
        * 0.00001
    )

    micro["BOX2"] = (
        population
        * synthetic_fiber_ratio
        * hand_wash_freq
        * hand_load
        * hand_ef
        * 0.00001
    )

    micro["total"] = micro["BOX1"] + micro["BOX2"]

    # =========================
    # 4. Environmental fate
    # =========================
    micro["BOX3"] = RTC23 * micro["BOX2"]
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
        micro["BOX4"] * RTC413
        + micro["BOX6"] * RTC613
        + micro["BOX7"] * RTC713
        + micro["BOX8"] * RTC813
        + micro["BOX9"] * RTC913
    )

    micro["BOX14"] = (
        micro["BOX3"] * RTC314
        + micro["BOX11"] * RTC1114
    )

    micro["BOX15"] = (
        micro["BOX13"] * RTC1315
        + micro["BOX14"] * RTC1415
    )

    micro["BOX16"] = (
        micro["BOX13"] * RTC1316
        + micro["BOX15"] * RTC1516
    )

    micro["BOX17"] = micro["BOX14"] * RTC1417
    micro["BOX18"] = micro["BOX15"] * RTC1518

    values_dict = {
        "MFs": micro["total"],
        "machine_MFs": micro["BOX1"],
        "hand_MFs": micro["BOX2"],
        "MFs_soil": micro["BOX17"],
        "MFs_ocean": micro["BOX16"],
        "MFs_river": micro["BOX18"],
        "MFs_environment": micro["BOX16"] + micro["BOX17"] + micro["BOX18"]
    }

    row_result = []

    for indicator in indicator_columns:
        row_result.extend(stat(values_dict[indicator]))

    city_result.iloc[i] = row_result


city_result = add_base_info(city_result, data_input)
write_stat_sheets(city_result, output_file)

print("MFs Monte Carlo results completed.")
print(f"Results saved to: {output_file}")