import datetime
from functools import singledispatch

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import kstest, pearsonr


@singledispatch
def calculate_auto_correlation_for_column(
    df, column, datetime_col="datetime", shifts=None
):
    """Generate auto-correlation values for different time windows.

    Note: input DataFrame should be sorted by the datetime column!

    Args:
        df (pd.DataFrame or pd.DataFrame): Input DataFrame.
        column (str): Column with values to use for calculation of correlation.
        datetime_col (str, optional): Column with datetime values.
        shifts (dict, optional): Be default, correlation will be calculated
            for hour, half_day, day, week and half_year. The `shifts` argument can be
            use to specify custom periods. The input is dictionary with name as key
            and number of rows to use (shift) as value.

    Returns:
        DataFrame with auto-correlation result.
    """
    return pl.DataFrame()


@calculate_auto_correlation_for_column.register
def _(df: pl.DataFrame, column: str, datetime_col="datetime", shifts=None):
    per_hour = int(datetime.timedelta(seconds=3600) / df[datetime_col].diff().mode()[0])

    default_shifts = {
        "hour": per_hour,
        "half_day": per_hour * 12,
        "day": per_hour * 24,
        "week": per_hour * 24 * 7,
        "half_year": per_hour * 24 * 7 * 26,
    }

    shifts = default_shifts if shifts is None else shifts
    result = {}
    for time_delta, delta in shifts.items():
        tmp = df.select(column).with_columns(
            pl.col(column).shift(delta).alias(time_delta)
        )
        nrows = tmp.shape[0] - delta
        result[time_delta] = pearsonr(
            tmp[column].tail(nrows).fill_null(0),
            tmp[time_delta].tail(nrows).fill_null(0),
        )[0]

    return pl.DataFrame(result)


@calculate_auto_correlation_for_column.register
def _(df: pd.DataFrame, column: str, datetime_col="datetime", shifts=None):
    include_index = isinstance(df.index, pd.DatetimeIndex)
    return calculate_auto_correlation_for_column(
        pl.from_pandas(df, include_index=include_index),
        column=column,
        datetime_col=datetime_col,
        shifts=shifts,
    ).to_pandas()


@singledispatch
def calculate_auto_correlation_for_dataframe(df, datetime_col, shifts):
    """Calculate auto-correlation values for all columns in a DataFrame.

    Args:
        df (DataFrame or LazyFrame): Input DataFrame.
        datetime_col (str, optional): Column with datetime values.
        shifts (dict, optional): Be default, correlation will be calculated
            for hour, half_day, day, week and half_year. The `shifts` argument can be
            use to specify custom periods. The input is dictionary with name as key
            and number of rows to use (shift) as value.

    Returns:
        DataFrame with auto-correlation values.
    """
    raise NotImplementedError("Unknown type for df")


@calculate_auto_correlation_for_dataframe.register
def _(df: pl.LazyFrame | pl.DataFrame, datetime_col="datetime", shifts=None):
    df = df.sort(datetime_col)
    df = df.collect() if isinstance(df, pl.LazyFrame) else df
    columns = df.select(pl.exclude(datetime_col)).columns

    return pl.concat(
        [calculate_auto_correlation_for_column(df, col) for col in columns]
    )


@calculate_auto_correlation_for_dataframe.register
def _(df: pd.DataFrame, datetime_col="datetime", shifts=None):
    include_index = isinstance(df.index, pd.DatetimeIndex)
    return calculate_auto_correlation_for_dataframe(
        pl.from_pandas(df, include_index=include_index),
        datetime_col=datetime_col,
        shifts=shifts,
    ).to_pandas()


def calculate_auto_correlation(
    dfs: dict[str, pd.DataFrame | pl.DataFrame | pl.LazyFrame],
    datetime_col="datetime",
    shifts=None,
):
    """Calculate auto-correlation values for all columns in a DataFrame.

    Args:
        dfs (dict): Input with name (`str`) as key and DataFrame or LazyFrame as value.
        datetime_col (str, optional): Column with datetime values.
        shifts (dict, optional): Be default, correlation will be calculated
            for hour, half_day, day, week and half_year. The `shifts` argument can be
            use to specify custom periods. The input is dictionary with name as key
            and number of rows to use (shift) as value.

    Returns:
        DataFrame with auto-correlation values.
    """
    fmt = (
        "pandas"
        if np.any([isinstance(df, pd.DataFrame) for df in dfs.values()])
        else "polars"
    )

    result = [
        calculate_auto_correlation_for_dataframe(
            df,
            datetime_col=datetime_col,
            shifts=shifts,
        )
        for df in dfs.values()
    ]

    result = [
        pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df for df in result
    ]

    corr_metrics = pl.concat(
        [
            df.unpivot(
                value_name="correlation", variable_name="time_delta"
            ).with_columns(pl.lit(name).alias("name"))
            for name, df in zip(dfs.keys(), result)
        ]
    )

    if fmt == "pandas":
        return corr_metrics.to_pandas()

    return corr_metrics


def plot_autocorrelation_stats(df: pd.DataFrame | pl.DataFrame):
    """CDF plot of the auto-correlation results.

    Args:
        df: df (DataFrame): Input DataFrame, output of`calculate_auto_correlation()`.
    """
    g = sns.FacetGrid(df, col="time_delta", hue="name")
    g.map(sns.ecdfplot, "correlation")
    g.add_legend()


@singledispatch
def pairwise_autocorrelation_kstest(df, a: str, b: str):
    """Pairwise Kolmogorov-Smirnov test of the auto-correlation resuls.

    Test the distribution of correlation values between two data sets in the input
    DataFrame `df`.

    Args:
        df: df (DataFrame): Input DataFrame, output of `calculate_auto_correlation()`.
        a (str): Name of data set to compare.
        b (str): Name of other data set to compare.

    Return:
        DataFrame with test results.
    """
    return df


@pairwise_autocorrelation_kstest.register
def _(df: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    return pl.concat(
        [
            pl.DataFrame(
                kstest(
                    part_df.filter(pl.col("name") == a)["correlation"]
                    .drop_nans()
                    .drop_nulls(),
                    part_df.filter(pl.col("name") == b)["correlation"]
                    .drop_nans()
                    .drop_nulls(),
                )
            )
            .transpose()
            .rename({"column_0": "statistic", "column_1": "p_value"})
            .with_columns(time_delta=pl.lit(part_df["time_delta"][0]))
            for part_df in df.partition_by("time_delta")
        ]
    )


@pairwise_autocorrelation_kstest.register
def _(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return pairwise_autocorrelation_kstest(pl.from_pandas(df), a=a, b=b).to_pandas()
