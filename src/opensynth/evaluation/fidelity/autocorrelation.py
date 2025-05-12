import datetime

import polars as pl
import seaborn as sns
from scipy.stats import kstest, pearsonr


def calculate_auto_correlation_for_column(df: pl.DataFrame, column: str):
    per_hour = int(
        datetime.timedelta(seconds=3600) / df["datetime"].diff().mode()[0]
    )

    shifts = {
        "hour": per_hour,
        "half_day": per_hour * 12,
        "day": per_hour * 24,
        "week": per_hour * 24 * 7,
        "half_year": per_hour * 24 * 7 * 26,
    }

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


def calculate_single_auto_correlation(df: pl.LazyFrame | pl.DataFrame):
    df = df.sort("datetime")
    df = df.collect() if isinstance(df, pl.LazyFrame) else df
    columns = df.select(pl.exclude("datetime")).columns

    return pl.concat(
        [calculate_auto_correlation_for_column(df, col) for col in columns]
    )


def calculate_auto_correlation(dfs: dict[str, pl.LazyFrame]):
    corr_metrics = pl.concat(
        [
            calculate_single_auto_correlation(df)
            .unpivot(value_name="correlation", variable_name="time_delta")
            .with_columns(pl.lit(name).alias("name"))
            for name, df in dfs.items()
        ]
    )
    return corr_metrics


def plot_autocorrelation_stats(df: pl.DataFrame):
    g = sns.FacetGrid(df, col="time_delta", hue="name")
    g.map(sns.ecdfplot, "correlation")
    g.add_legend()


def pairwise_autocorrelation_kstest(
    df: pl.DataFrame, a: str, b: str
) -> pl.DataFrame:
    return pl.concat(
        [
            pl.DataFrame(
                kstest(
                    part_df.filter(pl.col("name") == a)[
                        "correlation"
                    ].drop_nans(),
                    part_df.filter(pl.col("name") == b)[
                        "correlation"
                    ].drop_nans(),
                )
            )
            .transpose()
            .rename({"column_0": "statistic", "column_1": "p_value"})
            .with_columns(time_delta=pl.lit(part_df["time_delta"][0]))
            for part_df in df.partition_by("time_delta")
        ]
    )
