# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic DER augmentation: EV charging and rooftop PV.

EULP baseline homes have no EV charging or PV. A RECS-calibrated
share of homes gets each DER injected synthetically:

- EV: cold-climate Level 2 charging events (see config.EV_*),
  arrival-time and energy draws per home-day, more energy on cold
  days. Charging past midnight is truncated at the day boundary
  (days are modelled independently downstream).
- PV: PVWatts-normalised hourly generation shapes scaled by system
  size and subtracted from load. Net load can go negative.

All functions are pure and seeded: same inputs and seed give the
same output.
"""

import logging

import numpy as np
import polars as pl
from scipy.stats import truncnorm

from opensynth.datasets.new_england import config

logger = logging.getLogger(__name__)

# PV adoption is concentrated in single-family homes; multi-family
# and mobile homes get a reduced relative propensity.
PV_NON_SINGLE_FAMILY_WEIGHT = 0.25

_SINGLE_FAMILY_ARCHETYPES = (0, 1)


def assign_der_flags(
    df_homes: pl.DataFrame,
    df_shares: pl.DataFrame,
    seed: int = config.SAMPLING_SEED,
) -> pl.DataFrame:
    """
    Assign has_ev / has_pv flags to homes from RECS adoption shares.

    EV flags are uniform within a state. PV flags are weighted
    toward single-family archetypes while preserving the state-level
    expected share.

    Args:
        df_homes (pl.DataFrame): One row per home with columns ID,
            state, archetype.
        df_shares (pl.DataFrame): Output of recs.der_shares (state,
            pv_share, ev_share).
        seed (int): RNG seed.

    Returns:
        pl.DataFrame: df_homes with has_ev and has_pv columns (Int8).
    """
    rng = np.random.default_rng(seed)
    df = df_homes.join(df_shares, on="state", how="left").sort("ID")

    pv_weight = np.where(
        df["archetype"].is_in(_SINGLE_FAMILY_ARCHETYPES).to_numpy(),
        1.0,
        PV_NON_SINGLE_FAMILY_WEIGHT,
    )
    # Rescale weights per state so the expected share is preserved
    df = df.with_columns(pl.Series("pv_weight", pv_weight))
    df = df.with_columns(
        (
            pl.col("pv_share")
            * pl.col("pv_weight")
            / pl.col("pv_weight").mean().over("state")
        )
        .clip(0.0, 1.0)
        .alias("pv_prob")
    )

    n = len(df)
    has_ev = rng.random(n) < df["ev_share"].to_numpy()
    has_pv = rng.random(n) < df["pv_prob"].to_numpy()
    logger.info(
        f"🔌 DER flags: {int(has_ev.sum())} EV homes, "
        f"{int(has_pv.sum())} PV homes of {n}"
    )
    return df.drop(
        "pv_weight", "pv_prob", "pv_share", "ev_share"
    ).with_columns(
        pl.Series("has_ev", has_ev.astype(np.int8)),
        pl.Series("has_pv", has_pv.astype(np.int8)),
    )


def _interval_expr() -> pl.Expr:
    """15-minute interval-of-day index (0-95) from DateTime."""
    return (
        pl.col("DateTime").dt.hour() * 4 + pl.col("DateTime").dt.minute() // 15
    ).alias("interval")


def inject_ev(
    df_long: pl.DataFrame, seed: int = config.SAMPLING_SEED
) -> pl.DataFrame:
    """
    Add EV charging energy to flagged homes' 15-minute readings.

    Per home-day with has_ev=1: charge with probability
    EV_DAILY_CHARGE_PROB, arrival drawn from a truncated normal,
    energy from a normal (scaled up on days with temp_bin <= 2),
    delivered at EV_CHARGE_KW until exhausted or midnight.

    Args:
        df_long (pl.DataFrame): 15-minute rows with columns ID,
            DateTime, kwh, temp_bin, has_ev.
        seed (int): RNG seed.

    Returns:
        pl.DataFrame: Same rows with EV energy added to kwh.
    """
    rng = np.random.default_rng(seed)
    df = df_long.with_columns(
        _interval_expr(), pl.col("DateTime").dt.date().alias("date")
    )

    home_days = (
        df.filter(pl.col("has_ev") == 1)
        .group_by(["ID", "date"])
        .agg(pl.col("temp_bin").first())
        .sort(["ID", "date"])
    )
    n = len(home_days)
    if n == 0:
        return df_long

    charging = rng.random(n) < config.EV_DAILY_CHARGE_PROB
    lo, hi = config.EV_ARRIVAL_WINDOW_HOURS
    a = (lo - config.EV_ARRIVAL_MEAN_HOUR) / config.EV_ARRIVAL_STD_HOURS
    b = (hi - config.EV_ARRIVAL_MEAN_HOUR) / config.EV_ARRIVAL_STD_HOURS
    arrival_h = truncnorm.rvs(
        a,
        b,
        loc=config.EV_ARRIVAL_MEAN_HOUR,
        scale=config.EV_ARRIVAL_STD_HOURS,
        size=n,
        random_state=rng,
    )
    energy = rng.normal(
        config.EV_ENERGY_MEAN_KWH, config.EV_ENERGY_STD_KWH, n
    ).clip(min=0.5)
    is_cold = home_days["temp_bin"].to_numpy() <= 2
    energy = np.where(is_cold, energy * config.EV_WINTER_ENERGY_FACTOR, energy)
    energy = np.where(charging, energy, 0.0)

    # Spread energy across consecutive intervals at the charge rate
    kwh_per_interval = config.EV_CHARGE_KW / 4.0
    start = np.floor(arrival_h * 4).astype(int)
    max_slots = int(np.ceil(energy.max() / kwh_per_interval))
    rows = []
    for k in range(max_slots):
        remaining = energy - k * kwh_per_interval
        active = remaining > 0
        interval = start + k
        in_day = interval <= 95  # truncate at midnight
        mask = active & in_day
        if not mask.any():
            continue
        add = np.minimum(remaining[mask], kwh_per_interval)
        rows.append(
            pl.DataFrame(
                {
                    "ID": home_days["ID"].to_numpy()[mask],
                    "date": home_days["date"].to_numpy()[mask],
                    "interval": interval[mask],
                    "ev_kwh": add,
                }
            )
        )
    if not rows:
        return df_long

    additions = pl.concat(rows).with_columns(pl.col("interval").cast(pl.Int64))
    df = (
        df.join(additions, on=["ID", "date", "interval"], how="left")
        .with_columns(
            (pl.col("kwh") + pl.col("ev_kwh").fill_null(0.0)).alias("kwh")
        )
        .drop("ev_kwh", "interval", "date")
    )
    logger.info(
        f"🚗 Injected {float(additions['ev_kwh'].sum()):.1f} kWh of EV "
        f"charging across {int(charging.sum())} home-days"
    )
    return df


def apply_pv(
    df_long: pl.DataFrame,
    df_shape: pl.DataFrame,
    seed: int = config.SAMPLING_SEED,
) -> pl.DataFrame:
    """
    Subtract PV generation from flagged homes' readings (net load).

    Each has_pv home gets a system size drawn from config.PV_SIZES_KW
    and the PVWatts shape for its state's reference site. Net load
    can go negative during export.

    Args:
        df_long (pl.DataFrame): 15-minute rows with columns ID,
            DateTime, kwh, state, has_pv.
        df_shape (pl.DataFrame): Normalised shapes with columns site,
            month, hour, kw_per_kw (AC output per kW capacity).
        seed (int): RNG seed.

    Returns:
        pl.DataFrame: Same rows with PV generation subtracted.
    """
    rng = np.random.default_rng(seed)
    pv_homes = df_long.filter(pl.col("has_pv") == 1)["ID"].unique().sort()
    if len(pv_homes) == 0:
        return df_long

    sizes = pl.DataFrame(
        {
            "ID": pv_homes,
            "pv_kw": rng.choice(config.PV_SIZES_KW, size=len(pv_homes)),
        }
    )
    df = df_long.with_columns(
        pl.col("DateTime").dt.month().alias("month"),
        pl.col("DateTime").dt.hour().alias("hour"),
        pl.col("state").replace_strict(config.PV_STATE_SHAPE).alias("site"),
    )
    df = (
        df.join(sizes, on="ID", how="left")
        .join(df_shape, on=["site", "month", "hour"], how="left")
        .with_columns(
            (
                pl.col("kwh")
                - (pl.col("pv_kw") * pl.col("kw_per_kw") / 4.0).fill_null(0.0)
            ).alias("kwh")
        )
        .drop("month", "hour", "site", "pv_kw", "kw_per_kw")
    )
    logger.info(f"☀️ Applied PV to {len(pv_homes)} homes")
    return df
