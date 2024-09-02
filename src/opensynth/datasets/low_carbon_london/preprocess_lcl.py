# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from opensynth.datasets.datasets_utils import NoiseFactory, NoiseType

logger = logging.getLogger(__name__)


def get_current_month_end(df: pd.DataFrame, date_col="dt"):
    """
    Get the month end of current month.
    Note: Pandas offsets.MonthEnd()
    returns the date of the following month instead of the current month.

    E.g. if dt = 31st Jan, offsets.MonthEnd returns 28th Feb
    instead of 31st Jan. To work around this, we offset by -1 days
    to get previous day, then do offsets.MonthEnd()

    This is faster than offsets.MonthEnd().rollforward(ts):
    - This method doesn't work on array, only works on raw Timestamp object,
    which will require for loops (or .apply) which is super slow on big data
    - More efficient to perform this in a vectorised manner by offset
    with -1 days then use offsets.MonthEnd().

    Args:
        df (Pandas dataframe): Input pandas dataframe
        date_col (str, optional): Name of date column. Defaults to "dt".

    Returns:
        pd.DataFrame: Output dataframe with the "month_end" column
    """
    df["month_end"] = df[date_col] + pd.offsets.Day(-1)
    df["month_end"] = df["month_end"] + pd.offsets.MonthEnd()
    df["month_end"] = df["month_end"].dt.date
    return df


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load LCL data from csv

    Args:
        files_path (str): Folder containing CSV files

    Returns:
        pd.DataFrame: LCL dataset
    """
    logger.info(f"🚛 Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df = df.rename(columns={"KWH/hh (per half hour) ": "kwh"})
    return df


def extract_date_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse date features from DateTime column.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Output dataframe with date features
    """
    logger.info("📅 Extracting date features")
    df["dt"] = pd.to_datetime(df["DateTime"])
    df["date"] = df["dt"].dt.date.astype(str)
    df["month"] = df["dt"].dt.month.astype(int)
    df["week"] = df["dt"].dt.to_period("W-MON").dt.start_time
    df = get_current_month_end(df, date_col="dt")
    df["month_max"] = pd.to_datetime(df["month_end"]).dt.day.astype(int)
    df["day"] = df["dt"].dt.day.astype(int)
    df["dayofweek"] = df["dt"].dt.dayofweek.astype(int)
    df["hour"] = df["dt"].dt.hour.astype(int)
    df["minute"] = df["dt"].dt.minute.astype(int)
    return df


def parse_settlement_period(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse settlement periods from hour and minute columns

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Output dataframe with settlement period column
    """
    logger.info("🕰 Parsing Settlement Period")

    def _get_settlement_offset(minute_value):
        if minute_value >= 30:
            return 1
        return 0

    df_out = df.copy()
    df_out["settlement_offset"] = df_out["minute"].apply(
        _get_settlement_offset
    )
    df_out["settlement_period"] = (
        df_out["hour"] * 2 + df_out["settlement_offset"] + 1
    )
    df_out = df_out.drop(columns=["settlement_offset"])

    return df_out


def drop_dupes_and_replace_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to drop duplicated readings and replace missing readings with 0.0

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Output dataframe
    """
    logger.info("🗑 Dropping dupes and filling nulls with 0")
    df_out = df.copy()
    df_out = df_out.sort_values(
        by=["LCLid", "date", "settlement_period"], ascending=True
    )
    df_out = df_out.drop_duplicates(
        subset=["LCLid", "date", "settlement_period"], keep="last"
    )
    df_out["kwh"] = df_out["kwh"].replace("Null", np.float64())
    df_out["kwh"] = df_out["kwh"].astype(float)
    return df_out


def filter_missing_kwh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop dates where we don't have full 48 readings
    for a given LCLid and date.

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    logger.info("🔍 Filtering missing kwh readings")
    id_col = ["LCLid"]
    merge_cols = id_col + ["date"]
    df_group = df.groupby(merge_cols)[["kwh"]].count().reset_index()
    df_group["required_len"] = 48  # 48 hh readings

    df_full_data = df_group.query("required_len==kwh")  # Has all required data
    df_out = df_full_data[merge_cols].merge(df, on=merge_cols, how="inner")
    return df_out


def pack_smart_meter_data_into_arrays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pack smart meter data into 48-dimensional arrays.
    Note: LCL dataset are all given in UTC. We should expect
    48-readings per day.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Output dataframe
    """
    logger.info("👝 Packing time series into arrays")
    df_out = df.copy()
    df_out = df_out.sort_values(
        by=["LCLid", "date", "settlement_period"], ascending=True
    )
    groupby_cols = ["LCLid", "stdorToU", "month_end", "month"]
    groupby_cols = groupby_cols + ["dayofweek", "day", "date"]

    df_out = pd.DataFrame(
        df_out.groupby(groupby_cols)["kwh"]
        .agg(lambda x: x.tolist())
        .reset_index()
    )
    return df_out


def get_mean_and_std(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the mean and standard deviation of the dataset

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        Tuple[float, float]: Mean and standard deviation
    """
    logger.info("📊 Calculating mean and std")
    mean = np.mean(df["kwh"])
    std = np.std(df["kwh"])
    return mean, std


def create_outliers(
    df: pd.DataFrame, mean: float, mean_factor: int = 20
) -> pd.DataFrame:
    """
    Function to generate outliers based on gaussian and gamma distribution.
    Noise is generated based on a mean of 20 times population mean
    as descibed in the "Defining Good" paper.

    Args:
        df (pd.DataFrame): Input dataframe to sample rows from
        mean (float): Dataset population mean

    Returns:
        pd.DataFrame: Dataframe consisting of noisy outliers
    """

    gaussian_generator = NoiseFactory(
        noise_type=NoiseType.GAUSSIAN,
        mean=mean,
        scale=1.0,
        mean_factor=mean_factor,
        size=(50, 48),
    )

    gamma_generator = NoiseFactory(
        noise_type=NoiseType.GAMMA,
        mean=mean,
        scale=1.0,
        mean_factor=mean_factor,
        size=(50, 48),
    )
    logger.info(
        "🎲 Generating unseen outliers with mean:"
        f"{mean:.4f} and mean_factor: {mean_factor:.4f}"
    )
    df_gaussian_noise = gaussian_generator.inject_noise(df)
    df_gamma_noise = gamma_generator.inject_noise(df)

    df_noise = pd.concat([df_gaussian_noise, df_gamma_noise])
    return df_noise


def preprocess_pipeline(file_path: Path, out_path: Path):

    df = load_data(file_path)
    df = extract_date_features(df)
    df = parse_settlement_period(df)
    df = drop_dupes_and_replace_nulls(df)
    df = filter_missing_kwh(df)

    mean, stdev = get_mean_and_std(df)
    df = pack_smart_meter_data_into_arrays(df)

    df_noise = create_outliers(df, mean)

    os.makedirs(out_path, exist_ok=True)
    df.to_csv(f"{out_path}/lcl_data.csv", index=False)
    df_noise.to_csv(f"{out_path}/outliers.csv", index=False)

    mean_std_dict = {"mean": mean, "stdev": stdev}
    with open(f"{out_path}/mean_std.csv", "w") as f:
        w = csv.DictWriter(f, mean_std_dict.keys())
        w.writeheader()
        w.writerow(mean_std_dict)


def preprocess_lcl_data():

    SOURCE_DIR = "data/raw"
    OUT_DIR = "data/processed"
    preprocess_pipeline(
        file_path=f"{SOURCE_DIR}/historical/train.csv",
        out_path=f"{OUT_DIR}/historical/train",
    )
    preprocess_pipeline(
        file_path=f"{SOURCE_DIR}/historical/holdout.csv",
        out_path=f"{OUT_DIR}/historical/holdout",
    )
    preprocess_pipeline(
        file_path=f"{SOURCE_DIR}/future/train.csv",
        out_path=f"{OUT_DIR}/future/train",
    )
    preprocess_pipeline(
        file_path=f"{SOURCE_DIR}/future/holdout.csv",
        out_path=f"{OUT_DIR}/future/holdout",
    )


if __name__ == "__main__":
    preprocess_lcl_data()
