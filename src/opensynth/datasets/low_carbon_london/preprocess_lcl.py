# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
    Load data from csv

    Args:
        file_path (Path): Path to the csv file

    Returns:
        pd.DataFrame: dataset
    """
    logger.info(f"🚛 Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df


def format_data(
    df: pd.DataFrame,
    datetime_col: str,
    kwh_col: str,
    id_col: str,
    utc: bool = True,
    datetime_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from csv

    Args:
        df (pd.DataFrame): Dataset
        datetime_col (str): Name of the datetime column
        kwh_col (str): Name of the kWh column
        id_col (str): Name of the household ID column
        utc (bool): Whether the datetime is in UTC
        datetime_format (str, optional): Format of the datetime column. If
            None, will try to infer the format. Defaults to None.

    Returns:
        pd.DataFrame: dataset
    """

    logger.info("🧹 Formatting data")
    df.rename(
        columns={datetime_col: "DateTime", kwh_col: "kwh", id_col: "ID"},
        inplace=True,
    )
    df["DateTime"] = pd.to_datetime(
        df["DateTime"], utc=utc, format=datetime_format
    )
    df["kwh"] = df["kwh"].replace("Null", np.nan)  # Replace "Null" with np.nan
    df["kwh"] = df["kwh"].astype(float)
    df["ID"] = df["ID"].astype(str)
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


def drop_dupes_and_nulls(
    df: pd.DataFrame, drop_nulls: bool = True
) -> pd.DataFrame:
    """
    Function to drop duplicated readings and replace missing readings with 0.0

    Args:
        df (pd.DataFrame): Input dataframe
        drop_nulls (bool): Whether to drop rows with NaN kwh values. If False,
            will replace NaN kwh values with 0.0
    Returns:
        pd.DataFrame: Output dataframe
    """
    logger.info("🗑 Dropping dupes")
    df_out = df.copy()
    df_out = df_out.sort_values(
        by=["ID", "date", "settlement_period"], ascending=True
    )
    df_out = df_out.drop_duplicates(
        subset=["ID", "date", "settlement_period"], keep="last"
    )
    if drop_nulls:
        logger.info("🗑 Dropping nulls")
        df_out = df_out.dropna(subset="kwh")
    else:
        logger.info("🗑 Filling nulls with 0.0")
        df_out["kwh"] = df_out["kwh"].fillna(0.0)
    return df_out


def filter_missing_kwh(
    df: pd.DataFrame, time_resolution: str = "half_hourly"
) -> pd.DataFrame:
    """
    Drop dates where we don't have full 48 readings if half-hourly data,
    or 24 readings if hourly data, for a given ID and date.

    Args:
        df (pd.DataFrame): dataset
        time_resolution (str): Time resolution of the data. Allowed values:
            "half_hourly", "hourly".

    Returns:
        pd.DataFrame: filtered dataset
    """
    logger.info("🔍 Filtering missing kwh readings")
    merge_cols = ["ID", "date"]
    df_group = df.groupby(merge_cols)[["kwh"]].count().reset_index()

    if time_resolution == "half_hourly":
        required_len = 48  # 48 hh readings
    elif time_resolution == "hourly":
        required_len = 24  # 24 h readings
    else:
        raise ValueError(
            f"time_resolution must be 'half_hourly' or 'hourly', \
        got {time_resolution}"
        )
    df_group["required_len"] = required_len

    df_full_data = df_group.query("required_len==kwh")  # Has all required data
    df_out = df_full_data[merge_cols].merge(df, on=merge_cols, how="inner")

    if len(df_out) == 0:
        raise ValueError(
            "No data left after filtering days with missing kWh readings"
        )
    return df_out


def pack_smart_meter_data_into_arrays(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Pack smart meter data into 48- or 24-dimensional arrays,
    depending on whether half-hourly or hourly data.
    Note: LCL dataset are all given in UTC. We should expect
    48-readings per day.

    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (List[str]): List of feature columns to include in
            processed dataset.

    Returns:
        pd.DataFrame: Output dataframe
    """
    logger.info("👝 Packing time series into arrays")
    df_out = df.copy()
    df_out = df_out.sort_values(
        by=["ID", "date", "settlement_period"], ascending=True
    )
    groupby_cols = ["ID", "month_end", "month"] + feature_cols
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
    df: pd.DataFrame, time_resolution: str, mean: float, mean_factor: int = 20
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

    if time_resolution == "half_hourly":
        n = 48
    elif time_resolution == "hourly":
        n = 24
    else:
        raise ValueError("time_resolution must be 'half_hourly' or 'hourly'")

    gaussian_generator = NoiseFactory(
        noise_type=NoiseType.GAUSSIAN,
        mean=mean,
        scale=1.0,
        mean_factor=mean_factor,
        size=(50, n),
    )

    gamma_generator = NoiseFactory(
        noise_type=NoiseType.GAMMA,
        mean=mean,
        scale=1.0,
        mean_factor=mean_factor,
        size=(50, n),
    )
    logger.info(
        "🎲 Generating unseen outliers with mean:"
        f"{mean:.4f} and mean_factor: {mean_factor:.4f}"
    )
    df_gaussian_noise = gaussian_generator.inject_noise(df)
    df_gamma_noise = gamma_generator.inject_noise(df)

    df_noise = pd.concat([df_gaussian_noise, df_gamma_noise])
    return df_noise


def preprocess_pipeline(
    file_path: Path,
    out_path: Path,
    datetime_col: str = "DateTime",
    kwh_col: str = "kwh",
    id_col: str = "ID",
    utc: bool = True,
    datetime_format: Optional[str] = None,
    time_resolution: str = "half_hourly",
    feature_cols: List[str] = ["stdorToU"],
    drop_nulls: bool = True,
):
    """Preprocess the raw data for Faraday training and evaluation.
    Pipeline includes:
    - Data cleaning
    - Extraction of time features (day of week, month of year)
    - Injection of outliers into dataset
    - Calculation of mean and std for normalisation in Faraday
    - Re-structuring the dataset so that each row corresponds to a daily load
        profile.

    Args:
        file_path (Path): Path to the raw CSV file
        out_path (Path): Path to save the processed data
        datetime_col (str, optional): datetime column. Defaults to "DateTime".
        kwh_col (str, optional): kWh column. Defaults to "kwh".
        id_col (str, optional): Household ID column. Defaults to "ID".
        utc (bool, optional): Whether the datetime is in UTC. Defaults to True.
        datetime_format (str, optional): Format of the datetime column. If
            None, will try to infer the format. Defaults to None.
        time_resolution (str, optional): Time resolution of the data. Allowed
            values: "half_hourly", "hourly". Defaults to "half_hourly".
        feature_cols (List[str], optional): List of feature columns to include.
            Defaults to ["stdorToU"].
        drop_nulls (bool): Whether to drop rows with NaN kwh values. If False,
            will replace NaN kwh values with 0.0
    """

    df = load_data(file_path)
    df = format_data(df, datetime_col, kwh_col, id_col, utc, datetime_format)
    df = extract_date_features(df)
    df = parse_settlement_period(df)
    df = drop_dupes_and_nulls(df, drop_nulls)
    df = filter_missing_kwh(df, time_resolution)

    mean, stdev = get_mean_and_std(df)
    df = pack_smart_meter_data_into_arrays(df, feature_cols)

    df_noise = create_outliers(df, time_resolution, mean)

    os.makedirs(out_path, exist_ok=True)
    df.to_csv(f"{out_path}/data.csv", index=False)
    df_noise.to_csv(f"{out_path}/outliers.csv", index=False)

    mean_std_dict = {"mean": mean, "stdev": stdev}
    with open(f"{out_path}/mean_std.csv", "w") as f:
        w = csv.DictWriter(f, mean_std_dict.keys())
        w.writeheader()
        w.writerow(mean_std_dict)


def preprocess_data(
    data_dir: str,
    datetime_col: str = "DateTime",
    kwh_col: str = "kwh",
    id_col: str = "ID",
    utc: bool = True,
    datetime_format: Optional[str] = None,
    time_resolution: str = "half_hourly",
    feature_cols: List[str] = ["stdorToU"],
    drop_nulls: bool = True,
):

    SOURCE_DIR = f"{data_dir}/raw"
    OUT_DIR = f"{data_dir}/processed"
    preprocess_pipeline(
        file_path=Path(f"{SOURCE_DIR}/historical/train.csv"),
        out_path=Path(f"{OUT_DIR}/historical/train"),
        datetime_col=datetime_col,
        kwh_col=kwh_col,
        id_col=id_col,
        utc=utc,
        datetime_format=datetime_format,
        time_resolution=time_resolution,
        feature_cols=feature_cols,
        drop_nulls=drop_nulls,
    )
    preprocess_pipeline(
        file_path=Path(f"{SOURCE_DIR}/historical/holdout.csv"),
        out_path=Path(f"{OUT_DIR}/historical/holdout"),
        datetime_col=datetime_col,
        kwh_col=kwh_col,
        id_col=id_col,
        utc=utc,
        datetime_format=datetime_format,
        time_resolution=time_resolution,
        feature_cols=feature_cols,
        drop_nulls=drop_nulls,
    )
    preprocess_pipeline(
        file_path=Path(f"{SOURCE_DIR}/future/train.csv"),
        out_path=Path(f"{OUT_DIR}/future/train"),
        datetime_col=datetime_col,
        kwh_col=kwh_col,
        id_col=id_col,
        utc=utc,
        datetime_format=datetime_format,
        time_resolution=time_resolution,
        feature_cols=feature_cols,
        drop_nulls=drop_nulls,
    )
    preprocess_pipeline(
        file_path=Path(f"{SOURCE_DIR}/future/holdout.csv"),
        out_path=Path(f"{OUT_DIR}/future/holdout"),
        datetime_col=datetime_col,
        kwh_col=kwh_col,
        id_col=id_col,
        utc=utc,
        datetime_format=datetime_format,
        time_resolution=time_resolution,
        feature_cols=feature_cols,
        drop_nulls=drop_nulls,
    )
