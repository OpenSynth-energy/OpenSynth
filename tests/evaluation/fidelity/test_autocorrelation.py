from datetime import datetime

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_allclose

from opensynth.evaluation.fidelity.autocorrelation import (
    calculate_auto_correlation,
    calculate_auto_correlation_for_column,
    calculate_auto_correlation_for_dataframe,
)


@pytest.fixture
def test_dataframe_half_hour():
    df = pl.DataFrame(
        {
            "datetime": pl.Series(
                pl.datetime_range(
                    datetime(2020, 1, 1, 0, 0),
                    datetime(2020, 2, 1, 0, 0),
                    eager=True,
                    interval="30m",
                )
            ),
            "value": range(1489),
        }
    ).with_columns(
        pl.when(pl.col("value").mod(2) == 0)
        .then(pl.col("value"))
        .otherwise(pl.col("value").mul(-1))
        .alias("neg")
    )

    return df


@pytest.fixture
def test_dataframe_half_hour_pandas(test_dataframe_half_hour):
    return test_dataframe_half_hour.to_pandas()


@pytest.fixture(scope="module")
def test_dataframe_quarterly():
    """ "DataFrame with 15-minute timesteps and high correlation with a week time-lag."""
    n_minutes = 15
    n_values = 60 // n_minutes * 24 * 7  # 1 week
    n_timesteps = 35041
    np.random.seed(1)
    a = np.random.random(n_values)
    df = pl.DataFrame(
        {
            "datetime": pl.Series(
                pl.datetime_range(
                    datetime(2020, 1, 1, 0, 0),
                    datetime(2020, 12, 31, 0, 0),
                    eager=True,
                    interval="15m",
                )
            ),
            "week_corr": np.hstack(
                [a for _ in range(n_timesteps // n_values + 1)]
            )[:n_timesteps],
        }
    )

    return df


@pytest.mark.parametrize(
    "df_fixture,column,shifts,expected",
    (
        (
            "test_dataframe_half_hour",
            "value",
            {"two": 2, "three": 3},
            {"two": 1.0, "three": 1},
        ),
        (
            "test_dataframe_half_hour",
            "neg",
            {"two": 2, "three": 3},
            {"two": 1.0, "three": -1.0},
        ),
        (
            "test_dataframe_half_hour",
            "value",
            None,
            {
                "hour": 1.0,
                "half_day": 1,
                "day": 1,
                "week": 1,
                "half_year": np.nan,
            },
        ),
        (
            "test_dataframe_half_hour",
            "neg",
            None,
            {
                "hour": 1.0,
                "half_day": 1,
                "day": 1,
                "week": 0.985,
                "half_year": np.nan,
            },
        ),
        (
            "test_dataframe_quarterly",
            "week_corr",
            None,
            {
                "hour": -0.004,
                "half_day": -0.04,
                "day": 0.015,
                "week": 1.0,
                "half_year": 1.0,
            },
        ),
    ),
)
def test_calculate_auto_correlation_for_column(
    df_fixture, column, shifts, expected, request
):
    """Test autocorrelation calculation."""
    df = request.getfixturevalue(df_fixture)
    result = calculate_auto_correlation_for_column(
        df, column, shifts=shifts
    ).to_dict(as_series=False)
    for name in expected:
        assert_almost_equal(
            result[name][0],
            expected[name],
            decimal=3,
            err_msg=f"{name} failed",
        )


@pytest.mark.parametrize(
    "df_fixture,column,shifts,expected",
    (
        (
            "test_dataframe_half_hour_pandas",
            "value",
            {"two": 2, "three": 3},
            {"two": 1.0, "three": 1},
        ),
        (
            "test_dataframe_half_hour_pandas",
            "neg",
            {"two": 2, "three": 3},
            {"two": 1.0, "three": -1.0},
        ),
        (
            "test_dataframe_half_hour_pandas",
            "value",
            None,
            {
                "hour": 1.0,
                "half_day": 1,
                "day": 1,
                "week": 1,
                "half_year": np.nan,
            },
        ),
        (
            "test_dataframe_half_hour_pandas",
            "neg",
            None,
            {
                "hour": 1.0,
                "half_day": 1,
                "day": 1,
                "week": 0.985,
                "half_year": np.nan,
            },
        ),
    ),
)
def test_calculate_auto_correlation_for_column_pandas(
    df_fixture, column, shifts, expected, request
):
    """Test autocorrelation calculation (pandas input)."""
    df = request.getfixturevalue(df_fixture)
    result = calculate_auto_correlation_for_column(
        df, column, shifts=shifts
    ).to_dict()
    for name in expected:
        assert_almost_equal(
            result[name][0],
            expected[name],
            decimal=3,
            err_msg=f"{name} failed",
        )


@pytest.mark.parametrize(
    "df_fixture,column,expected",
    (
        ("test_dataframe_half_hour", "two", [1, 1]),
        ("test_dataframe_half_hour", "three", [1, -1]),
        ("test_dataframe_half_hour_pandas", "two", [1, 1]),
        ("test_dataframe_half_hour_pandas", "three", [1, -1]),
    ),
)
def test_calculate_auto_correlation_for_dataframe(
    df_fixture, column, expected, request
):
    df = request.getfixturevalue(df_fixture)
    result = calculate_auto_correlation_for_dataframe(
        df, shifts={"two": 2, "three": 3}
    )
    assert_array_almost_equal(result[column], expected, decimal=3)


@pytest.mark.parametrize(
    "df_fixture",
    (
        "test_dataframe_half_hour", 
        "test_dataframe_half_hour_pandas",

    ),
)
def test_calculate_auto_correlation(df_fixture, request):
    df = request.getfixturevalue(df_fixture)

    result = calculate_auto_correlation({"df1":df, "df2":df})
    assert result.shape == (20,3)
    assert list(sorted(result['name'].unique())) == ["df1", "df2"]
    values = np.array(result['correlation'])
    assert_allclose(values[~np.isnan(values)], 1, atol=0.02)


