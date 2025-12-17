# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from typing import Optional

import polars as pl
import polars.selectors as cs

DATE_COLUMNS = ["date", "DATE"]
DATETIME_COLUMNS = ["datetime", "DATETIME"]


logger = logging.getLogger(__name__)


def infer_date_column(df: pl.DataFrame) -> str:
    """Return column name for a Date columns in input DataFrame.

    Returns the column name of a column in Date format, or a String column that
    matches a Date string. If the DataFrame contains only one matching column,
    this function will return that column name. If multiple columns match, it
    will return the column name that matches a canonical Date name, such as
    "DATUM". In all other cases the function will raise a ValueError().

    Args:
        df (pl.DataFrame): DataFrame.

    Returns:
        str: column name of a column in Date or Date-like format.

    Raises:
        ValueError: if no columns are in a Date-like format or multiple columns
            are in Date-like format and match a canonical name.

    """
    date_columns = df.select(pl.col(pl.Date)).columns
    date_columns = list(
        set(df.columns).intersection(DATE_COLUMNS).union(date_columns)
    )
    canonical_columns = set(DATE_COLUMNS).intersection(date_columns)

    match len(date_columns):
        case 0:
            raise ValueError(
                "No Date or Date-like columns found in DataFrame!"
            )
        case 1:
            return date_columns[0]
        case _ if len(canonical_columns) == 1:
            return list(canonical_columns)[0]
        case _:
            raise ValueError(
                "Multiple Date-like columns found with a matching canonical \
                name!"
            )


def semiwide_to_long(
    df: pl.DataFrame,
    on: Optional[list[str]] = None,
    date_col: Optional[str] = None,
    datetime_name: Optional[str] = None,
    value_name: Optional[str] = None,
) -> pl.DataFrame:
    """Convert polars DataFrame from semi-wide to long format.

    The semi-wide format is based on a split between date (rows) and time
    (columns). Therefore, this function will only work on a DateFrame with
    a Date column that contains dates and at least one timestamp column, by
    default in "%HH%mm" format.

    Args:
        df (polars.DataFrame): DataFrame in semi-wide wide format, containing
            DateTime-compatible column names.
        on (list, optional): Columns to use as timepoints. By default, all
            columns that match the pattern '[0-9][0-9][0-9][0-9]' will be used.
        date_col (str, optional): Column that contains the Date values. By
            default, a column that is in Date format, or that is a
            Date-compatible string, will be used, if there is only one column
            in that format. If there are multiple Date-compatible columns, but
            only one matches a canonical name such as DATUM, that column will
            be used. Otherwise, this method will fail, and the date_col needs
            to be explicitly specified.
        datetime_name (str, optional): Name for the DateTime column in the long
            DataFrame, "DATUM_TIJD" by default.
        value_name (str, optional): Name to give to the value column. Defaults
            to "value".

    Returns:
        polars.DataFrame in long format.

    """
    on = df.select(cs.matches(r"^\d\d\d\d$")).columns if on is None else on
    date_col = infer_date_column(df) if date_col is None else date_col
    datetime_name = (
        DATETIME_COLUMNS[0] if datetime_name is None else datetime_name
    )
    value_name = "value" if value_name is None else value_name

    if str(df.select(date_col).dtypes[0]) == "String":
        df = df.with_columns(pl.col(date_col).str.to_date().alias(date_col))

    long_df = (
        df.unpivot(
            index=df.select(pl.exclude(on)).columns,
            on=df.select(on).columns,
            value_name=value_name,
        )
        .with_columns(
            (
                pl.col(date_col).dt.strftime("%Y-%m-%d")
                + " "
                + pl.col("variable")
            )
            .str.to_datetime(time_unit="ns", time_zone="UTC")
            .alias(datetime_name),
        )
        .drop(date_col, "variable")
        .sort(datetime_name)
    )

    return long_df


def semiwide_to_wide(
    df: pl.DataFrame,
    on: Optional[list[str]] = None,
    date_col: Optional[str] = None,
    datetime_name: Optional[str] = None,
) -> pl.DataFrame:
    """Convert polars DataFrame from semi-wide to wide format.

    The semi-wide format is based on a split between date (rows) and time
    (columns). Therefore, this function will only work on a DateFrame with
    a Date column that contains dates and at least one timestamp column, by
    default in "%HH%mm" format.

    Args:
        df (polars.DataFrame): DataFrame in semi-wide wide format, containing
            DateTime-compatible column names.
        on (list, optional): Columns to use as timepoints. By default, all
            columns that match the pattern '[0-9][0-9][0-9][0-9]' will be used.
        date_col (str, optional): Column that contains the Date values. By
            default, a column that is in Date format, or that is a
            Date-compatible string, will be used, if there is only one column
            in that format. If there are multiple Date-compatible columns, but
            only one matches a canonical name such as DATUM, that column will
            be used. Otherwise, this method will fail, and the date_col needs
            to be explicitly specified.
        datetime_name (str, optional): Name for the DateTime column in the long
            DataFrame, "datetime" by default.

    Returns:
        polars.DataFrame in wide format.

    """
    date_col = infer_date_column(df) if date_col is None else date_col
    datetime_name = (
        DATETIME_COLUMNS[0] if datetime_name is None else datetime_name
    )

    return (
        semiwide_to_long(
            df, on=on, date_col=date_col, datetime_name=datetime_name
        )
        .with_columns(
            pl.col(datetime_name)
            .dt.strftime("%Y-%m-%d %H:%M")
            .alias(datetime_name)
        )
        .pivot(on=datetime_name, values="value", aggregate_function="first")
    )


def randomize_index_column(
    df: pl.DataFrame,
    index_col_name: str = "index",
    sample_col_name: str = "sample",
) -> pl.DataFrame:
    """Randomize an index column.

    Args:
        df (DataFrame): Input DataFrame.
        index_col_name (str): Name of index column.
        sample_col_name (str): Name of new column containing the randomized
            index.

    Returns:
        DataFrame with index column values randomized.
    """
    # Ensure the sample indices are random
    sample_idx = sorted(df[index_col_name].unique())

    # Create a new column that has the same indices randomized
    df = df.with_columns(
        pl.col(index_col_name)
        .replace(
            dict(zip(sample_idx, random.sample(sample_idx, k=len(sample_idx))))
        )
        .alias(sample_col_name)
    )

    # Remove original columns, if name is different
    df = (
        df.select(sample_col_name, pl.exclude(index_col_name, sample_col_name))
        if index_col_name != sample_col_name
        else df
    )

    return df
