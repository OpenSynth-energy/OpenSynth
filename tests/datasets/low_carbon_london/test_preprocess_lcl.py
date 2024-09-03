from datetime import datetime

import pandas as pd

from opensynth.datasets.low_carbon_london import preprocess_lcl
from tests.utils import df_test


class TestPreprocessLCL:

    df_date = preprocess_lcl.extract_date_features(df_test())
    df_settlement_period = preprocess_lcl.parse_settlement_period(df_date)
    df_drop_dupes = preprocess_lcl.drop_dupes_and_replace_nulls(
        df_settlement_period
    )
    df_drop_missing = preprocess_lcl.filter_missing_kwh(df_drop_dupes)

    def test_week(self):
        expected_week = pd.to_datetime(
            [
                datetime(2012, 10, 9),
                datetime(2012, 11, 13),
                datetime(2012, 12, 11),
                datetime(2012, 12, 11),
                datetime(2013, 1, 15),
                datetime(2013, 1, 15),
                datetime(2013, 1, 15),
                datetime(2013, 1, 15),
            ]
        )
        assert (self.df_date["week"] == expected_week).all()

    def test_month_end(self):
        expected_month_end = pd.to_datetime(
            [
                datetime(2012, 10, 31),
                datetime(2012, 11, 30),
                datetime(2012, 12, 31),
                datetime(2012, 12, 31),
                datetime(2013, 1, 31),
                datetime(2013, 1, 31),
                datetime(2013, 1, 31),
                datetime(2013, 1, 31),
            ]
        )
        assert (self.df_date["month_end"] == expected_month_end).all()

    def test_month_max(self):
        expected_month_max = [31, 30, 31, 31, 31, 31, 31, 31]
        assert (self.df_date["month_max"] == expected_month_max).all()

    def test_day_of_week(self):
        expected_day_of_week = [
            4,  # Friday
            1,  # Tuesday
            4,  # Friday
            4,  # Friday
            1,  # Tuesday
            1,  # Tuesday
            1,  # Tuesday
            1,  # Tuesday
        ]
        assert (self.df_date["dayofweek"] == expected_day_of_week).all()

    def test_parse_settlement_period(self):
        expected_settlement_period = [2, 3, 4, 4, 5, 6, 7, 8]
        assert (
            self.df_settlement_period["settlement_period"]
            == expected_settlement_period
        ).all()

    def test_drop_dupes_and_replace_nulls(self):
        assert len(self.df_drop_dupes) == 7
        assert self.df_drop_dupes["kwh"].sum() == 2.1

    def test_filter_missing_kwh(self):
        # test_df is filled with only 1 kwh per date reading
        # filter_missing_kwh checks that each date has 48 readings
        # and drop dates with < 48 readings
        assert len(self.df_drop_missing) == 0

    def test_pack_smart_meter_data_into_arrays(self):
        df_packed = preprocess_lcl.pack_smart_meter_data_into_arrays(
            self.df_drop_dupes
        )
        assert len(df_packed) == 4
        assert df_packed.query("LCLid=='MAC000002' and month==1 and day==15")[
            "kwh"
        ].values.tolist()[0] == [0.0, 0.4, 0.5, 0.6]
