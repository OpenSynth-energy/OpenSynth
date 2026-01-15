# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def load_lcl_data_by_year(
    fname: Path | str | None = None,
    year: int = 2013,
    fmt: Literal["pandas", "polars"] = "pandas",
) -> pd.DataFrame | pl.DataFrame:
    """Load LCL data for a specific year.

    Returns a DataFrame in wide format.
    The first column contains the timestamp.

    Args:
        fname (str or Path): Location of the `train.csv` data file.
        year (int): Year to load.

    Returns:
        pl.DataFrame with KWH/hh measurements.
    """
    fname = (
        Path(__file__).parents[0] / "./data/raw/historical/train.csv"
        if fname is None
        else Path(fname)
    )
    if not fname.exists():
        raise FileNotFoundError(
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
