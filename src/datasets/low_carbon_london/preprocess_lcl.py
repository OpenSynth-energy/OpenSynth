import glob
import logging

import pandas as pd
import tqdm

logging.basicConfig(level=logging.DEBUG)


def get_current_month_end(df: pd.DataFrame, date_col="dt"):
    """
    Get the month end of current month.
    Note: Pandas offsets.MonthEnd()
    returns the date of the following month instead of the current month.

    E.g. if dt = 31st Jan, offsets.MonthEnd returns 28th Feb instead of 31st Jan.
    To work around this, we offset by -1 days to get previous day,
    then do offsets.MonthEnd()

    This is faster than offsets.MonthEnd().rollforward(ts):
    - This method doesn't work on array, only works on raw Timestamp object,
    which will require for loops (or .apply) which is super slow on big data
    - More efficient to perform this in a vectorised manner by offset with -1 days
    then use offsets.MonthEnd().

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


def load_data(folder_path: str) -> pd.DataFrame:
    """
    Load all CSV files in a folder into a single pandas Dataframe

    Args:
        files_path (str): Folder containing CSV files

    Returns:
        pd.DataFrame: LCL dataset
    """
    all_files_path = f"{folder_path}/*.csv"
    logging.info(f"🚛 Loading data from {folder_path}")
    all_files = glob.glob(all_files_path)

    assert len(all_files) > 0, "Folder is empty!"
    df = pd.DataFrame()

    for file in tqdm(all_files):
        df = pd.concat([df, pd.read_csv(file)])

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
    print("📅 Extracting date features")
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
    print("🕰 Parsing Settlement Period")

    def _get_settlement_offset(minute_value):
        if minute_value >= 30:
            return 1
        return 0

    df_out = df.copy()
    df_out["settlement_offset"] = df_out["minute"].apply(_get_settlement_offset)
    df_out["settlement_period"] = df_out["hour"] * 2 + df_out["settlement_offset"] + 1
    df_out = df_out.drop(columns=["settlement_offset"])

    return df_out


def preprocess_pipeline(folder_path: str, out_path: str):

    df = load_data(folder_path)
    df = extract_date_features(df)
    df = parse_settlement_period(df)
