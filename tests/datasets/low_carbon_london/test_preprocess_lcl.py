from datetime import datetime

import pandas as pd

from src.datasets.low_carbon_london import preprocess_lcl


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
    ]
    dt = [
        "2012-10-12 00:30:00",
        "2012-11-13 01:00:00",
        "2012-12-14 01:30:00",
    ]
    kwh = [
        0.1,
        0.2,
        0.3,
    ]
    df = pd.DataFrame(
        {
            "LCLid": lcl_id,
            "DateTime": dt,
            "kwh": kwh,
        }
    )
    return df


class TestPreprocessLCL:

    df = preprocess_lcl.extract_date_features(df_test())

    def test_week(self):
        expected_week = pd.to_datetime(
            [
                datetime(2012, 10, 9),
                datetime(2012, 11, 13),
                datetime(2012, 12, 11),
            ]
        )
        assert (self.df["week"] == expected_week).all()

    def test_month_end(self):
        expected_month_end = pd.to_datetime(
            [
                datetime(2012, 10, 31),
                datetime(2012, 11, 30),
                datetime(2012, 12, 31),
            ]
        )
        assert (self.df["month_end"] == expected_month_end).all()

    def test_month_max(self):
        expected_month_max = [31, 30, 31]
        assert (self.df["month_max"] == expected_month_max).all()

    def test_day_of_week(self):
        expected_day_of_week = [
            4,
            1,
            4,
        ]  # Friday, Tuesday, Friday
        assert (self.df["dayofweek"] == expected_day_of_week).all()

    def test_parse_settlement_period(self):
        df = preprocess_lcl.parse_settlement_period(self.df)
        expected_settlement_period = [2, 3, 4]
        assert (df["settlement_period"] == expected_settlement_period).all()
