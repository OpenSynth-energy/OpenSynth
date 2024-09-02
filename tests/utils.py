import pandas as pd


def df_test() -> pd.DataFrame:
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
    kwh = [0.1, 0.2, 0.3, 0.3, "Null", 0.4, 0.5, 0.6]
    tariff = ["A", "A", "A", "A", "A", "A", "A", "A"]
    df = pd.DataFrame(
        {"LCLid": lcl_id, "DateTime": dt, "kwh": kwh, "stdorToU": tariff}
    )
    return df
