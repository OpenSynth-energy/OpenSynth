# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Download raw data for the New England dataset pipeline.

Sources (all anonymous HTTPS):
- NREL End-Use Load Profiles (ResStock AMY2018 release 2): per-state
  metadata and per-building 15-minute timeseries parquet files.
- EIA RECS 2020 state microdata CSV.
- NOAA GHCN-Daily 2018 TMAX/TMIN per state via the NCEI data service.

ISO-NE hourly demand is used for validation only and is not
redistributed. Download it manually (ISO-NE SMD requires a free
account, or use the EIA v2 API) and load it with
``load_isone_hourly``.
"""

import logging
import time
import urllib.error
from pathlib import Path

import polars as pl

from opensynth.datasets import datasets_utils
from opensynth.datasets.new_england import config

logger = logging.getLogger(__name__)

DOWNLOAD_RETRIES = 3
RETRY_WAIT_SECONDS = 5

METADATA_COLS = [
    "bldg_id",
    "weight",
    "in.state",
    "in.geometry_building_type_recs",
    "in.heating_fuel",
]


def _download_if_missing(url: str, out_path: Path) -> bool:
    """
    Download a file unless it already exists.

    Args:
        url (str): Source URL
        out_path (Path): Destination path

    Returns:
        bool: True if the file was downloaded, False if skipped
    """
    if out_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            datasets_utils.download_data(url, out_path)
            return True
        except (urllib.error.URLError, OSError):
            # Drop any partial file so skip-if-exists stays sound
            out_path.unlink(missing_ok=True)
            if attempt == DOWNLOAD_RETRIES:
                raise
            logger.warning(
                f"⚠️ Download failed (attempt {attempt}/"
                f"{DOWNLOAD_RETRIES}), retrying: {url}"
            )
            time.sleep(RETRY_WAIT_SECONDS * attempt)
    return True


def download_eulp_metadata(data_dir: str = "./data") -> None:
    """
    Download EULP baseline metadata parquet for each NE state.

    Args:
        data_dir (str): Data directory. Defaults to "./data".
    """
    out_dir = Path(data_dir) / "raw/new_england/metadata"
    for state in config.NE_STATES:
        url = config.EULP_METADATA_URL.format(state=state)
        name = f"{state}_baseline_metadata_and_annual_results.parquet"
        if _download_if_missing(url, out_dir / name):
            logger.info(f"⬇️ Downloaded EULP metadata for {state}")
        else:
            logger.info(f"⏭️ EULP metadata for {state} already present")


def load_eulp_metadata(data_dir: str = "./data") -> pl.DataFrame:
    """
    Load EULP metadata for all NE states with encoded labels.

    Args:
        data_dir (str): Data directory. Defaults to "./data".

    Returns:
        pl.DataFrame: One row per building with columns bldg_id,
        state, weight, archetype, heating_fuel.
    """
    out_dir = Path(data_dir) / "raw/new_england/metadata"
    frames = []
    for state in config.NE_STATES:
        name = f"{state}_baseline_metadata_and_annual_results.parquet"
        df = pl.read_parquet(out_dir / name, columns=METADATA_COLS)
        frames.append(df.with_columns(pl.lit(state).alias("state")))
    df_all = pl.concat(frames)
    return df_all.with_columns(
        pl.col("in.geometry_building_type_recs")
        .replace_strict(config.ARCHETYPE_ENCODING)
        .alias("archetype"),
        pl.col("in.heating_fuel")
        .replace_strict(config.HEATING_FUEL_ENCODING)
        .alias("heating_fuel"),
    ).select(
        "bldg_id",
        "state",
        "weight",
        "archetype",
        "heating_fuel",
        "in.geometry_building_type_recs",
        "in.heating_fuel",
    )


def download_eulp_timeseries(
    manifest: pl.DataFrame, data_dir: str = "./data"
) -> None:
    """
    Download per-building 15-minute timeseries for a manifest.

    The loop skips files that already exist, so an interrupted
    download resumes where it left off.

    Args:
        manifest (pl.DataFrame): Building manifest with columns
            state, bldg_id (from sampling.select_buildings).
        data_dir (str): Data directory. Defaults to "./data".
    """
    out_dir = Path(data_dir) / "raw/new_england/eulp"
    n_downloaded = 0
    n_skipped = 0
    for row in manifest.iter_rows(named=True):
        state, bldg_id = row["state"], row["bldg_id"]
        url = config.EULP_TIMESERIES_URL.format(state=state, bldg_id=bldg_id)
        if _download_if_missing(url, out_dir / f"{state}_{bldg_id}-0.parquet"):
            n_downloaded += 1
        else:
            n_skipped += 1
    logger.info(
        f"⬇️ EULP timeseries: {n_downloaded} downloaded, "
        f"{n_skipped} already present"
    )


def download_recs(data_dir: str = "./data") -> None:
    """
    Download the RECS 2020 public microdata CSV.

    Args:
        data_dir (str): Data directory. Defaults to "./data".
    """
    out = Path(data_dir) / "raw/new_england/recs/recs2020_public_v7.csv"
    if _download_if_missing(config.RECS_URL, out):
        logger.info("⬇️ Downloaded RECS 2020 microdata")
    else:
        logger.info("⏭️ RECS 2020 microdata already present")


def download_ghcn(data_dir: str = "./data") -> None:
    """
    Download GHCN-Daily TMAX/TMIN for each state's station.

    Args:
        data_dir (str): Data directory. Defaults to "./data".
    """
    out_dir = Path(data_dir) / "raw/new_england/ghcn"
    for state, station in config.GHCN_STATIONS.items():
        url = config.GHCN_DATA_URL.format(
            station=station, year=config.WEATHER_YEAR
        )
        name = f"{state}_{station}_{config.WEATHER_YEAR}.csv"
        if _download_if_missing(url, out_dir / name):
            logger.info(f"⬇️ Downloaded GHCN daily data for {state}")
        else:
            logger.info(f"⏭️ GHCN daily data for {state} already present")


def load_isone_hourly(csv_path: Path) -> pl.DataFrame:
    """
    Load a manually-downloaded ISO-NE hourly demand CSV.

    Accepts either ISO-NE SMD exports or EIA v2 API exports; the
    file must contain a timestamp column and a demand column.

    Args:
        csv_path (Path): Path to the hourly demand CSV.

    Returns:
        pl.DataFrame: Columns timestamp (datetime), demand_mwh.
    """
    df = pl.read_csv(csv_path)
    ts_col = next(
        c for c in df.columns if c.lower() in ("period", "timestamp", "date")
    )
    demand_col = next(
        c for c in df.columns if "demand" in c.lower() or "mw" in c.lower()
    )
    return df.select(
        pl.col(ts_col).str.to_datetime().alias("timestamp"),
        pl.col(demand_col).cast(pl.Float64).alias("demand_mwh"),
    ).sort("timestamp")


def get_ne_data(data_dir: str = "./data") -> None:
    """
    Download everything except per-building timeseries.

    Timeseries downloads need a building manifest first: run
    sampling.select_buildings on the metadata, then
    download_eulp_timeseries with the manifest.

    Args:
        data_dir (str): Data directory. Defaults to "./data".
    """
    download_eulp_metadata(data_dir)
    download_recs(data_dir)
    download_ghcn(data_dir)
