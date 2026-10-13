# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

random.seed(42)

logger = logging.getLogger(__name__)


def split_periods(
    df: pd.DataFrame,
    datetime_col: str,
    sample_fraction: float = 0.75,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split dataset into training vs holdout periods.

    Args:
        df (pd.DataFrame): dataset
        datetime_col (str): Name of the datetime ID column
        sample_fraction (float): Fraction of period to include in
        training set

    Returns:
        Tuple[List[datetime.date], List[datetime.date]]: List of training and holdout days as calendar date objects
    """
    logger.info("Splitting dataset into train and holdout windows")
    unique_days = sorted(set(df[datetime_col].dt.date))
    random.shuffle(unique_days)
    sample_size = int(len(unique_days) * sample_fraction)

    train_year_day = unique_days[:sample_size]
    holdout_year_day = unique_days[sample_size:]

    return train_year_day, holdout_year_day


def split_historical_future_periods(
    df: pd.DataFrame,
    datetime_col: str,
    historical_start: str,
    historical_end: str,
    future_start: str,
    future_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into:
     1) historical period
     2) future period
    These periods are used for TSTR evaluation.

    Args:
        df (pd.DataFrame): pd.DataFrame
        datetime_col (str): Name of the datetime column
        historical_start (str): Start date for historical period
        historical_end (str): End date for historical period
        future_start (str): Start date for future period
        future_end (str): End date for future period

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Historical and Future dataframe.
    """
    # Format the start and end dates to include time information
    historical_start = pd.Timestamp(historical_start).replace(
        hour=00, minute=00, second=00
    )
    historical_end = pd.Timestamp(historical_end).replace(
        hour=23, minute=59, second=59
    )
    future_start = pd.Timestamp(future_start).replace(
        hour=00, minute=00, second=00
    )
    future_end = pd.Timestamp(future_end).replace(
        hour=23, minute=59, second=59
    )

    historical_start_mask = df[datetime_col] >= historical_start
    historical_end_mask = df[datetime_col] <= historical_end
    tstr_historical_mask = historical_start_mask & historical_end_mask

    future_start_mask = df[datetime_col] >= future_start
    future_end_mask = df[datetime_col] <= future_end
    tstr_future_mask = future_start_mask & future_end_mask

    df_historical = df.loc[tstr_historical_mask]
    df_future = df.loc[tstr_future_mask]
    return df_historical, df_future


def split_data(
    data_dir: str,
    csv_filename: Path,
    sample_fraction: float = 0.75,
    id_col: str = "ID",
    kwh_col: str = "demand",
    datetime_col: str = "datetime",
    utc=True,
    datetime_format: Optional[str] = None,
    historical_start: str = "2019-01-01",
    historical_end: str = "2024-12-31",
    future_start: str = "2025-01-02",
    future_end: str = "2025-01-02",
) -> None:
    """
    Split the dataset 4 ways:
    1) Historical Train days data
    2) Historical Holdout days data
    3) Future Train days data
    4) Future Holdout days data

    Historical data is used for training generative models.
    Future data is used for Train-Synthetic-Test-Real (TSTR) evaluation (in a future version)

    Args:
        data_dir (str): Directory to store the processed data.
        csv_filename (Path): Path to the CSV file containing the raw data.
        sample_fraction (float): Fraction of households to include in the
            training set. Defaults to 0.75.
        id_col (str): Name of the household ID column. Defaults to "ID".
        kwh_col (str): Name of the kWh column. Defaults to "demand".
        datetime_col (str): Name of the datetime column. Defaults to
            "datetime".
        utc (bool): Whether to parse datetime as UTC. Defaults to False.
        datetime_format (str, optional): Format of the datetime column.
            Defaults to None.
        historical_start (str): Start date for historical data. Defaults to
            "2019-01-01".
        historical_end (str): End date for historical data. Defaults to
            "2024-12-31".
        future_start (str): Start date for future data. Defaults to
            "2025-01-02".
        future_end (str): End date for future data. Defaults to "2025-01-02".
    Returns:
        None
    """

    logger.info(f"👀 Reading data from: {csv_filename}")
    df = pd.read_csv(csv_filename)

    logger.info("🧹 Formatting data")
    df[datetime_col] = pd.to_datetime(
        df[datetime_col], utc=utc, format=datetime_format
    )
    df[kwh_col] = df[kwh_col].replace("Null", np.float64())
    df[kwh_col] = df[kwh_col].astype(float)
    df[id_col] = df[id_col].astype(str)

    logger.info("🖖 Spliting households into train and holdout")
    train_days, holdout_days = split_periods(
        df,
        datetime_col,
        sample_fraction=sample_fraction,
    )
    logger.info(f"Train len: {len(train_days)}")
    logger.info(f"Holdout len: {len(holdout_days)}")

    logger.info("📆 Splitting data into train and holdout period")
    df_history, df_future = split_historical_future_periods(
        df,
        datetime_col,
        historical_start,
        historical_end,
        future_start,
        future_end,
    )
    logger.info(f"History len: {len(df_history)}")
    logger.info(f"Future len: {len(df_future)}")

    date_series = df_history[datetime_col].dt.date
    df_historical_train = df_history[date_series.isin(train_days)]
    df_historical_holdout = df_history[date_series.isin(holdout_days)]

    logger.info("📦 Saving train and holdout data")
    historical_path = Path(f"{data_dir}/raw/historical")
    #future_path = Path(f"{data_dir}/raw/future")
    os.makedirs(historical_path, exist_ok=True)
    #os.makedirs(future_path, exist_ok=True)

    df_historical_train.to_csv(
        f"{historical_path}/train.csv",
        index=False,
    )
    df_historical_holdout.to_csv(
        f"{historical_path}/holdout.csv",
        index=False,
    )
    #df_future_train.to_csv(f"{future_path}/train.csv", index=False)
    #df_future_holdout.to_csv(f"{future_path}/holdout.csv", index=False)

    logger.info("👍 Done!")
