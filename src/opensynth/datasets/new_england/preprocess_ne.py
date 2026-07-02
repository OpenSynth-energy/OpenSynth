# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Preprocess EULP buildings into Faraday training data.

Melts per-building EULP parquet files to the LCL long schema,
joins conditioning labels and daily temperature bins, injects DER
augmentation, then reuses the low_carbon_london pipeline functions
to pack daily 96-interval load profiles.

EULP timestamps are period-ending in local standard time (EST): the
first reading of a year is 00:15 on 1 January and the last is 00:00
on 1 January of the next year. They are shifted to period-beginning
so each calendar day owns exactly its own 96 intervals. Timestamps
stay in local time (utc=False downstream).

Buildings are processed in per-state chunks: the full corpus is
~63M long rows, which does not fit comfortably in pandas on a
laptop; packed daily profiles are ~1/96th the size.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

from opensynth.datasets.low_carbon_london import (
    preprocess_lcl,
    split_households,
)
from opensynth.datasets.new_england import (
    config,
    der_augmentation,
    recs,
    weather,
)

logger = logging.getLogger(__name__)

EULP_KWH_COL = "out.electricity.total.energy_consumption"

# Labels joined onto the long rows before packing. month and
# dayofweek come from the packing step itself, so they are excluded
# here; deriving from config keeps the conditioning schema single-
# sourced.
NE_FEATURE_COLS = [
    c for c in config.FEATURE_COLS if c not in ("month", "dayofweek")
]

# Default PV shape table committed with the package (built by
# scripts/fetch_pv_shapes.py).
DEFAULT_PV_SHAPE_PATH = Path(__file__).parent / "resources/pv_shapes_ne.csv"


def melt_building(parquet_path: Path, building_id: str) -> pl.DataFrame:
    """
    Melt one EULP timeseries parquet to the LCL long schema.

    Shifts period-ending timestamps to period-beginning.

    Args:
        parquet_path (Path): Per-building EULP parquet file.
        building_id (str): Value for the ID column.

    Returns:
        pl.DataFrame: Columns ID, DateTime, kwh.
    """
    df = pl.read_parquet(parquet_path, columns=["timestamp", EULP_KWH_COL])
    return df.select(
        pl.lit(building_id).alias("ID"),
        (pl.col("timestamp") - pl.duration(minutes=15)).alias("DateTime"),
        pl.col(EULP_KWH_COL).cast(pl.Float64).alias("kwh"),
    )


def _load_state_long(
    manifest_state: pl.DataFrame, state: str, data_dir: str
) -> pl.DataFrame:
    """Melt all manifest buildings for one state into long rows."""
    eulp_dir = Path(data_dir) / "raw/new_england/eulp"
    frames = []
    for row in manifest_state.iter_rows(named=True):
        bldg_id = row["bldg_id"]
        path = eulp_dir / f"{state}_{bldg_id}-0.parquet"
        if not path.exists():
            logger.warning(f"⚠️ Missing timeseries file: {path}")
            continue
        frames.append(melt_building(path, f"{state}_{bldg_id}"))
    return pl.concat(frames)


def _pack_chunk(
    df_long: pl.DataFrame, drop_nulls: bool = True
) -> pd.DataFrame:
    """
    Run one long-format chunk through the LCL packing pipeline.

    Args:
        df_long (pl.DataFrame): Long rows with ID, DateTime, kwh and
            the NE feature columns.
        drop_nulls (bool): Passed through to drop_dupes_and_nulls.

    Returns:
        pd.DataFrame: Packed daily profiles (one row per home-day).
    """
    df = df_long.to_pandas()
    df = preprocess_lcl.extract_date_features(df)
    df = preprocess_lcl.parse_settlement_period(df, periods_per_hour=4)
    df = preprocess_lcl.drop_dupes_and_nulls(df, drop_nulls)
    df = preprocess_lcl.filter_missing_kwh(
        df, time_resolution="quarter_hourly"
    )
    return preprocess_lcl.pack_smart_meter_data_into_arrays(
        df, feature_cols=NE_FEATURE_COLS
    )


def _save_split(
    df_packed: pd.DataFrame,
    mean: float,
    stdev: float,
    out_path: Path,
) -> None:
    """Write data.csv, outliers.csv and mean_std.csv for one split."""
    os.makedirs(out_path, exist_ok=True)
    df_noise = preprocess_lcl.create_outliers(
        df_packed, "quarter_hourly", mean
    )
    df_packed.to_csv(out_path / "data.csv", index=False)
    df_noise.to_csv(out_path / "outliers.csv", index=False)
    with open(out_path / "mean_std.csv", "w") as f:
        writer = csv.DictWriter(f, ["mean", "stdev"])
        writer.writeheader()
        writer.writerow({"mean": mean, "stdev": stdev})


def preprocess_ne_data(
    data_dir: str = "./data",
    pv_shape_path: Optional[Path] = None,
    sample_fraction: float = 0.75,
    seed: int = config.SAMPLING_SEED,
    drop_nulls: bool = True,
) -> None:
    """
    Full preprocessing: melt, label, augment, split, pack, save.

    Outputs mirror the LCL layout:
    data/processed/new_england/{train,holdout}/{data,outliers,mean_std}.csv

    With a single weather year there is no historical/future TSTR
    split; only the household train/holdout split applies.

    Args:
        data_dir (str): Data directory.
        pv_shape_path (Path, optional): PVWatts shape CSV (site,
            month, hour, kw_per_kw). Defaults to the packaged
            resources/pv_shapes_ne.csv; if that is missing too, PV
            augmentation is skipped and has_pv is forced to 0 so the
            label stays honest.
        sample_fraction (float): Train household fraction.
        seed (int): RNG seed for DER assignment, augmentation and
            the household split.
        drop_nulls (bool): Drop rows with null kwh readings.
    """
    manifest = pl.read_csv(
        Path(data_dir) / "raw/new_england/building_manifest.csv"
    )
    manifest = manifest.with_columns(
        (pl.col("state") + "_" + pl.col("bldg_id").cast(pl.Utf8)).alias("ID")
    )

    df_recs = recs.load_recs(Path(data_dir) / config.RECS_CSV_RELPATH)
    df_flags = der_augmentation.assign_der_flags(
        manifest.select("ID", "state", "archetype", "heating_fuel"),
        recs.der_shares(df_recs),
        seed=seed,
    )
    df_temp = weather.build_temp_bin_table(data_dir).drop("tmean_c")

    if pv_shape_path is None and DEFAULT_PV_SHAPE_PATH.exists():
        pv_shape_path = DEFAULT_PV_SHAPE_PATH
    df_shape = None
    if pv_shape_path is not None:
        df_shape = pl.read_csv(pv_shape_path)
        logger.info(f"☀️ PV shapes: {pv_shape_path}")
    else:
        # Without shapes a has_pv=1 label would condition the model
        # on nothing; zero the flags so the packed labels stay honest
        logger.warning(
            "⚠️ No PV shape table available: skipping PV augmentation "
            "and forcing has_pv=0 for all homes"
        )
        df_flags = df_flags.with_columns(
            pl.lit(0).cast(pl.Int8).alias("has_pv")
        )

    # Household-level train/holdout split
    train_ids, holdout_ids = split_households.split_household_ids(
        manifest.to_pandas(), "ID", sample_fraction=sample_fraction, seed=seed
    )
    splits = {"train": set(train_ids), "holdout": set(holdout_ids)}
    logger.info(
        f"🖖 Split: {len(train_ids)} train / "
        f"{len(holdout_ids)} holdout households"
    )

    packed: dict[str, list[pd.DataFrame]] = {"train": [], "holdout": []}
    stats = {"train": [0.0, 0.0, 0], "holdout": [0.0, 0.0, 0]}

    for state in config.NE_STATES:
        manifest_state = manifest.filter(pl.col("state") == state)
        if manifest_state.is_empty():
            continue
        logger.info(f"🏗 Processing {state}")
        df_long = _load_state_long(manifest_state, state, data_dir)

        # Conditioning labels: integer state code + manifest labels
        df_long = df_long.join(
            df_flags.select(
                "ID", "archetype", "heating_fuel", "has_ev", "has_pv"
            ),
            on="ID",
            how="left",
        ).with_columns(
            pl.lit(config.STATE_ENCODING[state]).alias("state_code"),
            pl.col("DateTime").dt.date().alias("date"),
        )
        df_long = (
            df_long.join(
                df_temp.filter(pl.col("state") == state).drop("state"),
                on="date",
                how="left",
            )
            .drop("date")
            .rename({"state_code": "state"})
        )

        # DER augmentation on 15-minute rows
        df_long = der_augmentation.inject_ev(df_long, seed=seed)
        if df_shape is not None:
            df_aug = df_long.with_columns(
                pl.lit(state).alias("state_postal")
            ).rename({"state": "state_code", "state_postal": "state"})
            df_aug = der_augmentation.apply_pv(df_aug, df_shape, seed=seed)
            df_long = df_aug.rename(
                {"state": "state_postal", "state_code": "state"}
            ).drop("state_postal")

        for split_name, ids in splits.items():
            df_split = df_long.filter(pl.col("ID").is_in(list(ids)))
            if df_split.is_empty():
                continue
            chunk = _pack_chunk(df_split, drop_nulls)
            # Accumulate standardisation stats from the packed
            # output, so mean_std.csv describes exactly the rows
            # written to data.csv (raw long rows would count nulls
            # in n but not in the sum, and include readings from
            # incomplete days the packing step drops)
            arr = np.asarray(chunk["kwh"].to_list(), dtype=float)
            s = stats[split_name]
            s[0] += float(arr.sum())
            s[1] += float((arr**2).sum())
            s[2] += arr.size
            packed[split_name].append(chunk)

    out_root = Path(data_dir) / "processed/new_england"
    for split_name, frames in packed.items():
        df_packed = pd.concat(frames, ignore_index=True)
        total, total_sq, n = stats[split_name]
        mean = total / n
        stdev = (total_sq / n - mean**2) ** 0.5
        logger.info(
            f"📦 {split_name}: {len(df_packed)} daily profiles, "
            f"mean {mean:.4f} kWh, stdev {stdev:.4f}"
        )
        _save_split(df_packed, mean, stdev, out_root / split_name)
    logger.info("👍 Done!")
