# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Daily temperature conditioning from GHCN-Daily observations.

Daily mean temperature is (TMAX + TMIN) / 2 from one station per
state, binned into the integer temp_bin conditioning label.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from opensynth.datasets.new_england import config

logger = logging.getLogger(__name__)


def load_ghcn_daily(csv_path: Path) -> pl.DataFrame:
    """
    Load an NCEI daily-summaries CSV for one station.

    GHCN TMAX/TMIN values are tenths of degrees Celsius, sometimes
    padded with spaces.

    Args:
        csv_path (Path): NCEI data-service CSV with STATION, DATE,
            TMAX, TMIN columns.

    Returns:
        pl.DataFrame: Columns date, tmean_c (daily mean, degrees C).
    """
    df = pl.read_csv(
        csv_path, schema_overrides={"TMAX": pl.Utf8, "TMIN": pl.Utf8}
    )

    def _tenths(col: str) -> pl.Expr:
        return pl.col(col).str.strip_chars().cast(pl.Float64) / 10.0

    df = df.select(
        pl.col("DATE").str.to_date().alias("date"),
        ((_tenths("TMAX") + _tenths("TMIN")) / 2.0).alias("tmean_c"),
    )
    # A row with a blank TMAX or TMIN yields a null tmean_c; dropping
    # it here folds present-but-empty observations into the missing-
    # days warning instead of letting NaN reach np.digitize (which
    # would silently assign the hottest bin).
    n_null = df["tmean_c"].null_count()
    if n_null:
        logger.warning(
            f"⚠️ {csv_path.name}: dropping {n_null} rows with missing "
            "TMAX/TMIN"
        )
        df = df.drop_nulls("tmean_c")
    return df


def to_temp_bin(tmean_c: np.ndarray) -> np.ndarray:
    """
    Bin daily mean temperatures into integer conditioning labels.

    Bin 0 is below -15C, bin 9 above 25C, 5C steps between.

    Args:
        tmean_c (np.ndarray): Daily mean temperatures, degrees C.

    Returns:
        np.ndarray: Integer bins 0-9.
    """
    tmean_c = np.asarray(tmean_c, dtype=float)
    if np.isnan(tmean_c).any():
        # np.digitize(nan) returns the last bin, silently labelling a
        # missing observation as the hottest day of the year
        raise ValueError("tmean_c contains NaN values")
    return np.digitize(tmean_c, config.TEMP_BIN_EDGES_C)


def build_temp_bin_table(data_dir: str = "./data") -> pl.DataFrame:
    """
    Build the (state, date) -> temp_bin join table for all states.

    Args:
        data_dir (str): Data directory containing the GHCN CSVs
            downloaded by get_data.download_ghcn.

    Returns:
        pl.DataFrame: Columns state, date, tmean_c, temp_bin.
    """
    ghcn_dir = Path(data_dir) / "raw/new_england/ghcn"
    frames = []
    for state, station in config.GHCN_STATIONS.items():
        name = f"{state}_{station}_{config.WEATHER_YEAR}.csv"
        df = load_ghcn_daily(ghcn_dir / name)
        n_missing = 365 - len(df)
        if n_missing > 0:
            logger.warning(
                f"⚠️ {state}: {n_missing} days missing from GHCN data"
            )
        frames.append(df.with_columns(pl.lit(state).alias("state")))
    df_all = pl.concat(frames)
    return df_all.with_columns(
        pl.Series("temp_bin", to_temp_bin(df_all["tmean_c"].to_numpy())).cast(
            pl.Int64
        )
    ).select("state", "date", "tmean_c", "temp_bin")
