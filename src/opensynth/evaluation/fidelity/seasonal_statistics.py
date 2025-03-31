from typing import Literal

import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import kstest


def add_season(
    df: pl.LazyFrame, datetime_col: str = "datetime"
) -> pl.LazyFrame:
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
            .replace(
                {1: "winter", 2: "spring", 3: "summer", 4: "fall", 0: "fall"}
            )
        )
    )


def seasonal_stats(
    df: pl.LazyFrame | pl.DataFrame | pd.DataFrame,
    datetime_col: str = "datetime",
    high_low: Literal["high", "low"] = "high",
    quantile: float = 0.2,
) -> pl.DataFrame:
    """Calculate statistics per season."""
    # Convert diverse inputs to to pl.LazyFrame
    if isinstance(df, pd.DataFrame):
        include_index = isinstance(df.index, pd.DatetimeIndex)
        df = pl.from_pandas(df, include_index=include_index)

    df = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Determine correct quantile
    quantile = 1.0 - quantile if high_low == "high" else quantile

    # Add the season based on the datetime column
    with_season = add_season(df, datetime_col).collect()

    # Determine number of times the threshold is exceeded
    filter_condition = (
        with_season.select(pl.exclude("datetime", "season"))
        .quantile(quantile)
        .select(pl.all().repeat_by(with_season.shape[0]).flatten())
    )
    season_stats = (
        (
            with_season.select(pl.exclude("datetime", "season"))
            >= filter_condition
        )
        if high_low == "high"
        else (
            with_season.select(pl.exclude("datetime", "season"))
            <= filter_condition
        )
    )

    # Collect stats per season
    season_stats = (
        season_stats.with_columns(
            season=with_season.select("season")["season"]
        )
        .group_by("season")
        .sum()
    )

    return season_stats


def calculate_seasonal_stats(dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    result = [
        (
            seasonal_stats(df)
            .with_columns(
                pl.exclude("season").truediv(
                    df.select(pl.len()).collect().item()
                ),
                pl.lit(name).alias("name"),
            )
            .unpivot(index=["season", "name"])
        )
        for name, df in dfs.items()
    ]
    return pl.concat(result, how="vertical")


def print_seasonal_stats(df):
    print(
        df.pivot(
            on="name",
            index="season",
            values="value",
            aggregate_function="median",
        )
    )


def plot_seasonal_stats(df):
    sns.boxplot(
        data=df,
        hue="name",
        x="season",
        y="value",
        order=["winter", "spring", "summer", "fall"],
        fliersize=0,
    )


def pairwise_seasonal_kstest(df: pl.DataFrame, a: str, b: str) -> pl.Series:
    return pl.Series(
        [
            kstest(
                df.filter(pl.col("season") == season, pl.col("name") == a)[
                    "value"
                ],
                df.filter(pl.col("season") == season, pl.col("name") == b)[
                    "value"
                ],
            ).statistic
            for season in ["spring", "summer", "fall", "winter"]
        ]
    )
