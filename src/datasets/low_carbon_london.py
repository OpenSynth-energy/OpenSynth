import logging
import os
import random
from typing import List, Tuple

import pandas as pd

from src.datasets import datasets_utils

random.seed(42)
logging.basicConfig(level=logging.DEBUG)

LCL_URL = "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"  # noqa
FILE_NAME = "data/raw/lcl_full_data.zip"  # noqa
CSV_FILE_NAME = "data/raw/CC_LCL-FullData.csv"


def split_household_ids(
    df: pd.DataFrame, id_col: str, sample_size: int
) -> Tuple[List[str], List[str]]:
    """
    Split LCL id into training vs holdout households.

    Args:
        df (pd.DataFrame): LCL dataset
        id_col (str): ID column
        sample_size (int): Number of household ids in each split

    Returns:
        Tuple[List[str], List[str]]: List of training and holdout household ids
    """
    logging.info("Splitting households into train and holdout households")
    unique_ids = df[id_col].unique().tolist()
    random.shuffle(unique_ids)

    train_ids = unique_ids[:sample_size]
    holdout_ids = unique_ids[-sample_size:]

    return train_ids, holdout_ids


def split_historical_future_periods(
    df: pd.DataFrame, date_col: str = "DateTime"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits LCL dataset into:
     1) historical period (1st Jan 2012 to 31st Dec 2013)
     2) future period (after 1st Jan 2014)

     These periods are used for TSTR evaluation.

    Args:
        df (pd.DataFrame): pd.DataFrame
        date_col (str, optional): Name of date column. Defaults to "DateTime".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Historical and Future dataframe.
    """
    historical_start = df[date_col] >= "2012-01-01"
    historical_end = df[date_col] <= "2013-12-31"
    tstr_historical_mask = historical_start & historical_end

    future_start = df[date_col] >= "2014-01-01"
    future_end = df[date_col] <= "2014-12-31"
    tstr_future_mask = future_start & future_end

    df_historical = df.loc[tstr_historical_mask]
    df_future = df.loc[tstr_future_mask]
    return df_historical, df_future


def split_lcl_data(sample_size: int = 2000):
    """
    Split LCL dataset 4 ways:
    1) Historical Train household data
    2) Historical Holdout household data
    3) Future Train household data
    4) Future Holdout household data

    Historical data is used for training generative models.
    Future data is used for TSTR evaluation.

    Args:
        sample_size (int, optional): _description_. Defaults to 2000.
    """

    logging.debug(f"👀 Reading LCL data from: {FILE_NAME}")
    df = pd.read_csv(CSV_FILE_NAME)

    logging.debug("🖖 Spliting households into train and holdout")
    train_ids, holdout_ids = split_household_ids(
        df, id_col="LCLid", sample_size=sample_size
    )
    logging.debug(f"Train len: {len(train_ids)}")
    logging.debug(f"Holdout len: {len(holdout_ids)}")

    logging.debug("📆 Splitting data into train and holdout period")
    df_history, df_future = split_historical_future_periods(df)
    logging.debug(f"History len: {len(df_history)}")
    logging.debug(f"Future len: {len(df_future)}")

    df_historical_train = df_history[df_history["LCLid"].isin(train_ids)]
    df_future_train = df_future[df_future["LCLid"].isin(train_ids)]
    df_historical_holdout = df_history[df_history["LCLid"].isin(holdout_ids)]
    df_future_holdout = df_future[df_future["LCLid"].isin(holdout_ids)]

    logging.debug("📦 Saving train and holdout data")
    historical_path = "data/processed/historical"
    future_path = "data/processed/future"
    os.makedirs(historical_path, exist_ok=True)
    os.makedirs(future_path, exist_ok=True)
    df_historical_train.to_csv(f"{historical_path}/train.csv", index=False)
    df_historical_holdout.to_csv(f"{historical_path}/holdout.csv", index=False)
    df_future_train.to_csv(f"{future_path}/train.csv", index=False)
    df_future_holdout.to_csv(f"{future_path}/holdout.csv", index=False)

    logging.debug("👍 Done!")


def get_lcl_data(download: bool, split: bool, preprocess: bool):
    if download:
        datasets_utils.download_data(LCL_URL, FILE_NAME)
    if split:
        split_lcl_data(sample_size=2000)
    if preprocess:
        pass


if __name__ == "__main__":
    get_lcl_data(download=False, split=True, preprocess=False)
