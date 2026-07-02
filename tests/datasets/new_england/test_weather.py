from pathlib import Path

import numpy as np
import pytest

from opensynth.datasets.new_england import weather

FIXTURE = Path("tests/data/new_england/ghcn_toy.csv")


class TestLoadGhcnDaily:

    df = weather.load_ghcn_daily(FIXTURE)

    def test_tenths_of_degrees_parsed(self):
        # (-13.8 + -23.8) / 2 = -18.8
        assert self.df["tmean_c"].to_list()[0] == pytest.approx(-18.8)
        # (30.0 + 22.0) / 2 = 26.0
        assert self.df["tmean_c"].to_list()[2] == pytest.approx(26.0)

    def test_dates_parsed(self):
        assert str(self.df["date"].to_list()[0]) == "2018-01-01"


class TestToTempBin:

    def test_bin_edges(self):
        temps = np.array([-18.8, -15.0, -12.0, 0.0, 4.9, 5.0, 24.9, 26.0])
        bins = weather.to_temp_bin(temps)
        # Below -15 -> 0; -15 itself starts bin 1; 5.0 starts bin 5;
        # above 25 -> 9
        assert bins.tolist() == [0, 1, 1, 4, 4, 5, 8, 9]

    def test_fixture_bins(self):
        df = weather.load_ghcn_daily(FIXTURE)
        bins = weather.to_temp_bin(df["tmean_c"].to_numpy())
        # -18.8 -> 0, 0.0 -> 4, 26.0 -> 9, 10.0 -> 6
        assert bins.tolist() == [0, 4, 9, 6]
