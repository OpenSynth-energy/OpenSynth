from functools import singledispatch
from typing import Literal, cast

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import kstest


@singledispatch
def add_season(
    df,
    datetime_col,
):
    """Add a column with season to a DataFrame.

    The season is based on a datetime column. This column is `"datetime"` by default,
    but can be specified with the `datetime_col` argument.
    The season `"column"` will contain one of four values: "winter", "spring", "summer"
    or "fall".

    Args:
        df (DataFrame): Input DataFrame or LazyFrame.
        datetime_col (str, optional): Name of the column used as a datetime column
            ("datetime" by default).

    Returns:
        DataFrame with "season" column added.
    """
    return df


@add_season.register
def _(
    df: pl.LazyFrame | pl.DataFrame,
    datetime_col: str = "datetime",
) -> pl.LazyFrame | pl.DataFrame:
    return (
        df.with_columns(season=pl.col(datetime_col).dt.month() % 12 // 3 + 1)
        .with_columns(
            pl.when(
                pl.col(datetime_col).dt.month().is_in((3, 6, 9, 12))
                & (pl.col(datetime_col).dt.day() < 21)
            )
            .then(pl.col("season") - 1)
            .otherwise(pl.col("season"))
        )
        .with_columns(
            pl.col("season")
            .cast(str)
            .replace({1: "winter", 2: "spring", 3: "summer", 4: "fall", 0: "fall"})
        )
    )


@add_season.register
def _(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    return cast(
        pl.DataFrame, add_season(pl.from_pandas(df), datetime_col=datetime_col)
    ).to_pandas()


@singledispatch
def seasonal_peaks(
    df,
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
):
    """Calculate statistics on number of peaks per season.

    Args:
        df (DataFrame): Input DataFrame or LazyFrame.
        datetime_col (str, optional): Name of the column used as a datetime column
            ("datetime" by default). If `df` is a pandas DataFrame, `datetime_col` can also
            be the name of the index.
        high_low (str, optional): Either "high" or "low". Determines if the seasonal peaks are
            calculated for the high or for low peaks. Default is "high".
        quantile (float, option): Quantile used to determine peaks. Default is 0.2 / 0.8.

    Returns:
        DataFrame with statistics per season.
    """
    return df


@seasonal_peaks.register
def _(
    df: pl.DataFrame | pl.LazyFrame,
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
) -> pl.DataFrame:
    # Ensure we use the lower quantile
    quantile = min(quantile, 1 - quantile)
    # Reverse the quantile if we want to determine "high" peaks
    quantile = 1.0 - quantile if high_low == "high" else quantile

    # Add the season based on the datetime column
    with_season = add_season(df, datetime_col)

    # Determine number of times the threshold is exceeded
    filter_condition = (
        with_season.select(pl.exclude("datetime", "season"))
        .quantile(quantile)
        .select(pl.all().repeat_by(with_season.shape[0]).flatten())
    )

    season_stats = (
        (with_season.select(pl.exclude("datetime", "season")) >= filter_condition)
        if high_low == "high"
        else (with_season.select(pl.exclude("datetime", "season")) <= filter_condition)
    )

    # Collect stats per season
    season_stats = (
        season_stats.with_columns(season=with_season.select("season")["season"])
        .group_by("season")
        .sum()
    )

    return season_stats


@seasonal_peaks.register
def _(
    df: pl.LazyFrame,
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
) -> pl.DataFrame:
    return seasonal_peaks(
        df.collect(),
        datetime_col=datetime_col,
        high_low=high_low,
        quantile=quantile,
    )


@seasonal_peaks.register
def _(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Calculate statistics per season."""

    include_index = isinstance(df.index, pd.DatetimeIndex)
    return seasonal_peaks(
        pl.from_pandas(df, include_index=include_index),
        datetime_col=datetime_col,
        high_low=high_low,
        quantile=quantile,
    ).to_pandas()


def calculate_seasonal_peaks(
    dfs: dict[str, pd.DataFrame] | dict[str, pl.DataFrame | pl.LazyFrame],
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
):
    """Calculate statistics on number of peaks per season for multiple DataFrames.

    This function will calculate the number of peaks for all DataFrames in the `dfs`
    input dictionary.

    Args:
        dfs (dict): Input with name (`str`) as key and DataFrame or LazyFrame as value.
        datetime_col (str, optional): Name of the column used as a datetime column
            ("datetime" by default). If `df` is a pandas DataFrame, `datetime_col` can also
            be the name of the index.
        high_low (str, optional): Either "high" or "low". Determines if the seasonal peaks are
            calculated for the high or for low peaks. Default is "high".
        quantile (float, option): Quantile used to determine peaks. Default is 0.2 / 0.8.

    Returns:
        DataFrame with statistics per season for each data set.
    """
    fmt = (
        "pandas"
        if np.any([isinstance(df, pd.DataFrame) for df in dfs.values()])
        else "polars"
    )
    result = [
        seasonal_peaks(
            df,
            datetime_col=datetime_col,
            high_low=high_low,
            quantile=quantile,
        )
        for df in dfs.values()
    ]

    result = [
        pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df for df in result
    ]
    result = [
        (
            df.with_columns(
                pl.exclude("season").truediv(df.select(pl.len()).item()),
                pl.lit(name).alias("name"),
            ).unpivot(index=["season", "name"])
        )
        for name, df in zip(dfs.keys(), result)
    ]
    df = pl.concat(result, how="vertical")

    if fmt == "pandas":
        return df.to_pandas()

    return df


@singledispatch
def print_seasonal_stats(
    df,
    aggregate_function,
) -> None:
    """Print summary statistics of the seasonal statistics.

    Args:
        df: df (DataFrame): Input DataFrame or LazyFrame, output of
            `calculate_seasonal_peaks()`.
        aggregate_function (str or pl.Expr): Aggregate function to use, "median" by
            default.
    """


@print_seasonal_stats.register
def _(
    df: pl.DataFrame,
    aggregate_function: (
        Literal["min", "max", "first", "last", "sum", "mean", "median", "len"]
        | pl.Expr
        | None
    ) = "median",
) -> None:
    print(
        df.pivot(
            on="name",
            index="season",
            values="value",
            aggregate_function=aggregate_function,
        )
    )


@print_seasonal_stats.register
def _(
    df: pd.DataFrame,
    aggregate_function: (
        Literal["min", "max", "first", "last", "sum", "mean", "median", "len"]
        | pl.Expr
        | None
    ) = "median",
) -> None:
    print_seasonal_stats(pl.from_pandas(df), aggregate_function=aggregate_function)


def plot_seasonal_stats(df: pd.DataFrame | pl.DataFrame) -> None:
    """Boxplot of the number of peaks per season.

    Args:
        df: df (DataFrame): Input DataFrame or LazyFrame, output of
            `calculate_seasonal_peaks()`.
    """
    ax = sns.boxplot(
        data=df,
        hue="name",
        x="season",
        y="value",
        order=["winter", "spring", "summer", "fall"],
        fliersize=0,
    )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    sns.despine()


@singledispatch
def pairwise_seasonal_kstest(df, a: str, b: str):
    """Pairwise Kolmogorov-Smirnov test of the seasonal peaks.

    Test the distribution of peak number between two data sets in the input
    DataFrame `df`.

    Args:
        df: df (DataFrame): Input DataFrame or LazyFrame, output of
            `calculate_seasonal_peaks()`.
        a (str): Name of data set to compare.
        b (str): Name of other data set to compare.

    Return:
        DataFrame with test results.
    """
    return df


@pairwise_seasonal_kstest.register
def _(df: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    return pl.concat(
        [
            pl.DataFrame(
                kstest(
                    df.filter(pl.col("season") == season, pl.col("name") == a)["value"],
                    df.filter(pl.col("season") == season, pl.col("name") == b)["value"],
                )
            )
            .transpose()
            .rename({"column_0": "statistic", "column_1": "p_value"})
            .with_columns(season=pl.lit(season))
            for season in ["spring", "summer", "fall", "winter"]
        ]
    )


@pairwise_seasonal_kstest.register
def _(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return pairwise_seasonal_kstest(pl.from_pandas(df), a=a, b=b).to_pandas()
