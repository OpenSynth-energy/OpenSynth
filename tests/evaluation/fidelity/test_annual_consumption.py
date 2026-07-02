import numpy as np
import polars as pl
import pytest

from opensynth.evaluation.fidelity import annual_consumption


class TestAnnualKwhPerHome:

    def test_sums_per_home(self):
        df = pl.DataFrame(
            {
                "home_id": ["a", "a", "b", "b", "b"],
                "kwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        totals = annual_consumption.annual_kwh_per_home(df)
        assert totals.tolist() == [3.0, 12.0]

    def test_pandas_input(self):
        df = pl.DataFrame(
            {"home_id": ["a", "b"], "kwh": [10.0, 20.0]}
        ).to_pandas()
        totals = annual_consumption.annual_kwh_per_home(df)
        assert totals.tolist() == [10.0, 20.0]


class TestMeanPercentError:

    def test_exact_match_is_zero(self):
        assert annual_consumption.mean_percent_error(
            np.array([7000.0, 9000.0]), 8000.0
        ) == pytest.approx(0.0)

    def test_signed_error(self):
        assert annual_consumption.mean_percent_error(
            np.array([8800.0]), 8000.0
        ) == pytest.approx(10.0)
        assert annual_consumption.mean_percent_error(
            np.array([7200.0]), 8000.0
        ) == pytest.approx(-10.0)


class TestKsStatistic:

    def test_identical_distributions_score_zero(self):
        rng = np.random.default_rng(0)
        sample = rng.normal(8000, 1500, 500)
        assert annual_consumption.ks_statistic(
            sample, sample
        ) == pytest.approx(0.0)

    def test_disjoint_distributions_score_one(self):
        assert annual_consumption.ks_statistic(
            np.full(100, 1000.0), np.full(100, 9000.0)
        ) == pytest.approx(1.0)


class TestDecileTable:

    def test_identical_distributions_have_zero_errors(self):
        rng = np.random.default_rng(1)
        sample = rng.normal(8000, 1500, 1000)
        table = annual_consumption.decile_table(sample, sample)
        assert len(table) == 9
        assert table["percent_error"].abs().max() == pytest.approx(0.0)
