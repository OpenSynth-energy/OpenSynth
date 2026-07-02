from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def df_test_half_hourly() -> pd.DataFrame:
    """
    Test Dataframe

    Returns:
        pd.DataFrame: Test Dataframe
    """
    lcl_id = [
        "MAC000002",
        "MAC000002",
        "MAC000002",
        "MAC000002",
        "MAC000002",
        "MAC000002",
        "MAC000002",
        "MAC000002",
    ]
    dt = [
        "2012-10-12 00:30:00",
        "2012-11-13 01:00:00",
        "2012-12-14 01:30:00",
        "2012-12-14 01:30:00",
        "2013-01-15 02:00:00",
        "2013-01-15 02:30:00",
        "2013-01-15 03:00:00",
        "2013-01-15 03:30:00",
    ]
    kwh = [0.1, "Null", 0.3, 0.3, 0.4, 0.4, 0.5, 0.6]
    tariff = ["A", "A", "A", "A", "A", "A", "A", "A"]
    df = pd.DataFrame(
        {"LCLid": lcl_id, "DateTime": dt, "kwh": kwh, "stdorToU": tariff}
    )
    return df


def df_test_quarter_hourly() -> pd.DataFrame:
    """
    Test Dataframe

    Returns:
        pd.DataFrame: Test Dataframe
    """
    lcl_id = ["MAC000002"] * 192

    # start time
    start = datetime(2013, 1, 3, 0, 0, 0)
    # generate 192 timestamps, 15 minutes apart (2 full days)
    dt = [
        (start + timedelta(minutes=15 * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(192)
    ]

    kwh = np.random.rand(192).round(2).tolist()
    tariff = ["A"] * 192
    df = pd.DataFrame(
        {"LCLid": lcl_id, "DateTime": dt, "kwh": kwh, "stdorToU": tariff}
    )
    return df


def df_test_hourly() -> pd.DataFrame:
    """
    Test Dataframe

    Returns:
        pd.DataFrame: Test Dataframe
    """
    lcl_id = ["MAC000002"] * 48

    # start time
    start = datetime(2013, 1, 3, 0, 0, 0)
    # generate 48 timestamps, 1 hour apart
    dt = [
        (start + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(48)
    ]

    kwh = np.random.rand(48).round(2).tolist()
    tariff = ["A"] * 48
    df = pd.DataFrame(
        {"LCLid": lcl_id, "DateTime": dt, "kwh": kwh, "stdorToU": tariff}
    )
    return df
