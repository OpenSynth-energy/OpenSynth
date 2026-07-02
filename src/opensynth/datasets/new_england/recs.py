# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""RECS 2020 microdata: conditioning distributions and validation
references for New England.

Column codes (verified against recs2020_public_v7.csv):
- TYPEHUQ: 1 mobile home, 2 single-family detached, 3 single-family
  attached, 4 apartment 2-4 units, 5 apartment 5+ units
- FUELHEAT: 1 natural gas, 2 propane, 3 fuel oil/kerosene,
  5 electricity, 7 wood, -2 no heating
- SOLAR / EVCHRGHOME: 1 yes, 0 no, -2 not applicable
- NWEIGHT: household sampling weight
"""

import logging
from pathlib import Path

import polars as pl

from opensynth.datasets.new_england import config

logger = logging.getLogger(__name__)

RECS_COLS = [
    "state_postal",
    "NWEIGHT",
    "TYPEHUQ",
    "FUELHEAT",
    "KWH",
    "SOLAR",
    "EVCHRGHOME",
]

# RECS TYPEHUQ -> archetype encoding (config.ARCHETYPE_ENCODING order)
TYPEHUQ_TO_ARCHETYPE = {2: 0, 3: 1, 4: 2, 5: 2, 1: 3}

# RECS FUELHEAT -> heating fuel encoding. Wood (7) and no-heating
# (-2) map to other (4), matching the EULP Other Fuel / None bucket.
FUELHEAT_TO_HEATING_FUEL = {5: 0, 1: 1, 3: 2, 2: 3, 7: 4, -2: 4}


def load_recs(csv_path: Path) -> pl.DataFrame:
    """
    Load RECS microdata filtered to New England, with encoded labels.

    Args:
        csv_path (Path): Path to the RECS 2020 public microdata CSV.

    Returns:
        pl.DataFrame: One row per sampled household with columns
        state, nweight, archetype, heating_fuel, annual_kwh, has_pv,
        has_ev.
    """
    logger.info(f"🚛 Loading RECS microdata from {csv_path}")
    df = pl.read_csv(csv_path, columns=RECS_COLS)
    df = df.filter(pl.col("state_postal").is_in(config.NE_STATES))
    return df.select(
        pl.col("state_postal").alias("state"),
        pl.col("NWEIGHT").alias("nweight"),
        pl.col("TYPEHUQ")
        .replace_strict(TYPEHUQ_TO_ARCHETYPE)
        .alias("archetype"),
        pl.col("FUELHEAT")
        .replace_strict(FUELHEAT_TO_HEATING_FUEL, default=4)
        .alias("heating_fuel"),
        pl.col("KWH").cast(pl.Float64).alias("annual_kwh"),
        (pl.col("SOLAR") == 1).cast(pl.Int8).alias("has_pv"),
        (pl.col("EVCHRGHOME") == 1).cast(pl.Int8).alias("has_ev"),
    )


def joint_distribution(df_recs: pl.DataFrame) -> pl.DataFrame:
    """
    NWEIGHT-weighted joint distribution of
    state x archetype x heating_fuel.

    Shares are normalised within each state, so they sum to 1 per
    state and can drive per-state building allocation directly.

    Args:
        df_recs (pl.DataFrame): Output of load_recs.

    Returns:
        pl.DataFrame: Columns state, archetype, heating_fuel, share.
    """
    cell = df_recs.group_by(["state", "archetype", "heating_fuel"]).agg(
        pl.col("nweight").sum().alias("cell_weight")
    )
    return (
        cell.with_columns(
            (
                pl.col("cell_weight")
                / pl.col("cell_weight").sum().over("state")
            ).alias("share")
        )
        .drop("cell_weight")
        .sort(["state", "archetype", "heating_fuel"])
    )


def annual_kwh_summary(df_recs: pl.DataFrame) -> pl.DataFrame:
    """
    Weighted annual-kWh summary per state, for validation.

    Args:
        df_recs (pl.DataFrame): Output of load_recs.

    Returns:
        pl.DataFrame: Columns state, mean_kwh, median_kwh.
    """
    weighted_mean = (pl.col("annual_kwh") * pl.col("nweight")).sum() / pl.col(
        "nweight"
    ).sum()
    return (
        df_recs.group_by("state")
        .agg(
            weighted_mean.alias("mean_kwh"),
            pl.col("annual_kwh").median().alias("median_kwh"),
        )
        .sort("state")
    )


def der_shares(df_recs: pl.DataFrame) -> pl.DataFrame:
    """
    Weighted PV and home-EV-charging adoption shares per state.

    Drives assign_der_flags in the augmentation step.

    Args:
        df_recs (pl.DataFrame): Output of load_recs.

    Returns:
        pl.DataFrame: Columns state, pv_share, ev_share.
    """

    def _weighted_share(flag_col: str) -> pl.Expr:
        return (pl.col(flag_col) * pl.col("nweight")).sum() / pl.col(
            "nweight"
        ).sum()

    return (
        df_recs.group_by("state")
        .agg(
            _weighted_share("has_pv").alias("pv_share"),
            _weighted_share("has_ev").alias("ev_share"),
        )
        .sort("state")
    )
