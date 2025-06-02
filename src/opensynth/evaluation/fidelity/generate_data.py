import logging
from calendar import monthrange
from collections.abc import Generator
from datetime import date
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
import torch
from tqdm.auto import tqdm

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday import FaradayModel

DATE_COLUMNS = ["date", "DATE"]
DATETIME_COLUMNS = ["datetime", "DATETIME"]


logger = logging.getLogger(__name__)


def load_lcl_data_by_year(
    fname: Path | str | None = None,
    year: int = 2013,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load LCL data for a specific year.

    Returns a DataFrame in wide format. The first column contains the timestamp.

    ArgsL
        fname (str or Path): Location of the `train.csv` data file.
        year (int): Year to load.

    Returns:
        pl.DataFrame with KWH/hh measurements.
    """
    fname = (
        Path(__file__).parents[0] / "../../../../data/raw/historical/train.csv"
        if fname is None
        else Path(fname)
    )
    if not fname.exists():
        raise ValueError(
            "LCL dataset not found, "
            "please download it or supply correct path to train.csv"
        )

    logger.info(f"Loading LCL data from {str(fname.resolve())}...")
    lcl = (
        pl.scan_csv(
            fname,
            schema={
                "LCLid": pl.String,
                "stdorTtoU": pl.String,
                "DateTime": pl.String,
                "KWH/hh (per half hour)": pl.String,
            },
        )
        .with_columns(
            pl.col("KWH/hh (per half hour)")
            .str.strip_chars()
            .cast(pl.Float32, strict=False)
            .fill_null(0)
            .alias("kWH"),
            pl.col("DateTime")
            .str.slice(0, 16)
            .str.to_datetime()
            .alias("datetime"),
        )
        .collect()
        .unique()
        .pivot(on="LCLid", index="datetime", values="kWH")
        .sort("datetime")
        .fill_null(0)
    )

    logger.info(f"Selecting year {year}")
    lcl = lcl.filter(pl.col("datetime").dt.year() == year)

    if fmt == "pandas":
        return lcl.to_pandas()

    return lcl


def generate_synthetic_samples(
    model: FaradayModel,
    dm: LCLDataModule,
    n_samples: int,
    year: int = 2022,
    month: int | None = None,
) -> Generator[
    Tuple[date, float, float, np.typing.NDArray[np.float64]], None, None
]:
    """Generate Faraday samples for a specific month/year combination.

    Samples will be generated with a timestamp that fits the specified year and month.
    If month is not specified, it can be any month.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule): Data module.
        n_samples (int): Number of synthetic samples to generate.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.

    Yields:
        Tuple with datetime, month, day_of_week, generated sample values
    """
    if n_samples < 2:
        raise ValueError("n_samples must be higher than 1")

    sample_df = (
        pl.date_range(date(year, 1, 1), date(year, 12, 31), "1d", eager=True)
        .alias("datetime")
        .to_frame()
        .with_columns(
            pl.col("datetime").dt.weekday().alias("weekday"),
            pl.col("datetime").dt.month().alias("month"),
        )
    )

    gmm_samples = model.sample_gmm(n_samples)
    gmm_samples_reconstructed = dm.reconstruct_kwh(gmm_samples["kwh"])
    gmm_samples_reconstructed = torch.clip(gmm_samples_reconstructed, min=0)
    for torch_month, dayofweek, values in zip(
        gmm_samples["features"]["month"],
        gmm_samples["features"]["dayofweek"],
        gmm_samples_reconstructed,
    ):
        g_month = torch_month.numpy()[0]
        try:
            if month is None or g_month == month:
                yield (
                    sample_df.filter(
                        pl.col("weekday") == dayofweek.numpy()[0] + 1,
                        pl.col("month") == g_month,
                    ).sample(1)["datetime"][0],
                    g_month,
                    dayofweek.numpy()[0],
                    values.detach().numpy(),
                )

        except Exception as e:
            print(e)
            continue


def generate_synthetic_sample_df(
    model: FaradayModel,
    dm: LCLDataModule,
    n_samples: int,
    year: int = 2022,
    month: int | None = None,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific month/year combination.

    Samples will be generated with a timestamp that fits the specified year and month.
    If month is not specified, it can be any month.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule): Data module.
        n_samples (int): Number of synthetic samples to generate.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    df = pl.DataFrame(
        np.array(
            [
                (datetime, m, d, *values)
                for datetime, m, d, values in generate_synthetic_samples(
                    model, dm, n_samples, year=year, month=month
                )
            ]
        ).tolist(),
        schema={"date": pl.Date, "month": int, "dayofweek": int}
        | {
            d: float
            for d in [f"{i // 2:02d}{(i % 2) * 30:02d}" for i in range(48)]
        },
        orient="row",
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def generate_full_synthetic_month(
    model: FaradayModel,
    dm: LCLDataModule,
    year: int,
    month: int,
    n_samples: int = 2,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific month.

    Samples will be generated with a timestamp that fits the specified month.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule): Data module.
        year (int, optional): Year to use for timestamps.
        month (int, optional): Month (1-based) to use. If generated samples do not
            match the specified month, they will be discarded until enough samples
            are specified that do match.
        n_samples (int): Number of synthetic samples to generate.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    batch_size = 1000
    df = generate_synthetic_sample_df(
        model, dm, batch_size, year=year, month=month, fmt="polars"
    )

    while (
        df.group_by("date").len().min()["len"][0] < n_samples
        or len(df["date"].unique()) < monthrange(year, month)[1]
    ):
        df = pl.concat(
            (
                df,
                generate_synthetic_sample_df(
                    model,
                    dm,
                    batch_size,
                    year=year,
                    month=month,
                    fmt="polars",
                ),
            )
        )
    df = pl.concat(
        [p.sample(n_samples).with_row_index() for p in df.partition_by("date")]
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def generate_full_synthetic_year(
    model: FaradayModel,
    dm: LCLDataModule,
    year: int,
    n_samples: int = 2,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Generate DataFrame Faraday samples for a specific year.

    Samples will be generated with all timesteps for all months in the specified year.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule): Data module.
        year (int, optional): Year to use for timestamps.
        n_samples (int): Number of synthetic samples to generate.
        fmt (str, optional): Either "pandas" or "polars", default is "pandas".

    Returns:
        pl.DataFrame in wide format with datetime as first columns.
    """
    df = pl.concat(
        [
            generate_full_synthetic_month(
                model, dm, year, month, n_samples=n_samples, fmt="polars"
            )
            for month in tqdm(range(1, 13))
        ]
    )
    df = (
        semiwide_to_wide(
            df.select(pl.exclude("month", "dayofweek")),
            date_col="date",
            datetime_name="datetime",
        )
        .with_columns(pl.col("index").cast(str))
        .transpose(
            column_names="index", include_header=True, header_name="datetime"
        )
        .with_columns(pl.col("datetime").str.to_datetime())
    )

    if fmt == "pandas":
        return df.to_pandas()

    return df


def infer_date_column(df: pl.DataFrame) -> str:
    """Return column name for a Date columns in input DataFrame.

    Returns the column name of a column in Date format, or a String column that
    matches a Date string. If the DataFrame contains only one matching column,
    this function will return that column name. If multiple columns match, it will
    return the column name that matches a canonical Date name, such as "DATUM".
    In all other cases the function will raise a ValueError().

    Args:
        df (pl.DataFrame): DataFrame.

    Returns:
        str: column name of a column in Date or Date-like format.

    Raises:
        ValueError: if no columns are in a Date-like format or multiple columns are
        in Date-like format and match a canonical name.

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
                "Multiple Date-like columns found with a matching canonical name!"
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
        df (polars.DataFrame): DataFrame in semi-wide wide format, containing DateTime-
            compatible column names.
        on (list, optional): Columns to use as timepoints. By default, all columns that
            match the pattern '[0-9][0-9][0-9][0-9]' will be used.
        date_col (str, optional): Column that contains the Date values. By default,
            a column that is in Date format, or that is a Date-compatible string, will
            be used, if there is only one column in that format. If there are multiple
            Date-compatible, columns, but only one matches a canonical name such as
            DATUM, that column will be used. Otherwise, this method will fail, and the
            date_col needs to be explicitly specified.
        datetime_name (str, optional): Name for the DateTime column in the long
            DataFrame, "DATUM_TIJD" by default.
        value_name (str, optional): Name to give to the value column. Defaults to
            "value".

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
        date_col (str, optional): Column that contains the Date values. By default,
            a column that is in Date format, or that is a Date-compatible string,
            will be used, if there is only one column in that format. If there are
            multiple Date-compatible, columns, but only one matches a canonical name
            such as DATUM, that column will be used. Otherwise, this method will fail,
            and the date_col needs to be explicitly specified.
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
