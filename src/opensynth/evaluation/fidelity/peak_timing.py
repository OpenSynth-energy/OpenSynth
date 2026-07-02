# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

"""Peak-timing fidelity metric.

Compares the time-of-day of the seasonal aggregate demand peak in
synthetic profiles against a regional reference (e.g. ISO-NE hourly
system load). The synthetic aggregate is downsampled to hourly
resolution before comparison, since system references are hourly.

Note: system load includes commercial and industrial demand, so the
comparison is about peak *timing*, not magnitude or exact shape.
"""

from functools import singledispatch
from typing import Any, Sequence

import numpy as np
import pandas as pd
import polars as pl


def downsample_to_hourly(profile: np.ndarray) -> np.ndarray:
    """
    Sum a sub-hourly daily profile into 24 hourly values.

    Args:
        profile (np.ndarray): Daily profile whose length is a
            multiple of 24 (e.g. 96 quarter-hourly readings).

    Returns:
        np.ndarray: 24 hourly energy values.
    """
    if len(profile) % 24 != 0:
        raise ValueError(
            f"Profile length {len(profile)} is not a multiple of 24"
        )
    return np.asarray(profile).reshape(24, -1).sum(axis=1)


def synthetic_peak_hour(
    kwh: np.ndarray, months: np.ndarray, season_months: Sequence[int]
) -> float:
    """
    Hour of day of the aggregate synthetic peak for a season.

    Args:
        kwh (np.ndarray): Daily profiles [n_days, intervals].
        months (np.ndarray): Month (1-12) of each profile [n_days].
        season_months (Sequence[int]): Months forming the season.

    Returns:
        float: Peak hour of day (0-23).
    """
    mask = np.isin(months, season_months)
    if not mask.any():
        raise ValueError(f"No profiles in months {season_months}")
    mean_profile = kwh[mask].mean(axis=0)
    return float(np.argmax(downsample_to_hourly(mean_profile)))


@singledispatch
def reference_peak_hour(  # pragma: no cover
    df_reference: Any, season_months: Sequence[int]
) -> float:
    """
    Hour of day of the mean reference-load peak for a season.

    Args:
        df_reference (DataFrame): Hourly reference load with
            timestamp and demand_mwh columns.
        season_months (Sequence[int]): Months forming the season.

    Returns:
        float: Peak hour of day (0-23).
    """
    raise NotImplementedError(f"Unsupported input type: {type(df_reference)}")


@reference_peak_hour.register
def _(df_reference: pl.DataFrame, season_months: Sequence[int]) -> float:
    hourly = (
        df_reference.filter(
            pl.col("timestamp").dt.month().is_in(list(season_months))
        )
        .group_by(pl.col("timestamp").dt.hour().alias("hour"))
        .agg(pl.col("demand_mwh").mean())
        .sort("hour")
    )
    # Index the hour column: arg_max alone is a row position, which
    # only equals the hour when all 24 hours are present
    return float(hourly["hour"][hourly["demand_mwh"].arg_max()])


@reference_peak_hour.register
def _(df_reference: pd.DataFrame, season_months: Sequence[int]) -> float:
    return reference_peak_hour(pl.from_pandas(df_reference), season_months)


def peak_timing_delta_minutes(
    synthetic_hour: float, reference_hour: float
) -> float:
    """
    Absolute peak-timing difference in minutes, wrapping midnight.

    Args:
        synthetic_hour (float): Synthetic peak hour of day.
        reference_hour (float): Reference peak hour of day.

    Returns:
        float: Difference in minutes (0-720).
    """
    delta_hours = abs(synthetic_hour - reference_hour)
    delta_hours = min(delta_hours, 24 - delta_hours)
    return delta_hours * 60.0


def check_peak_within_tolerance(
    delta_minutes: float, tolerance_minutes: float = 60.0
) -> bool:
    """
    Validation criterion: peak timing within tolerance.

    Args:
        delta_minutes (float): Output of peak_timing_delta_minutes.
        tolerance_minutes (float): Allowed difference. Defaults to
            60 (the winter-peak criterion).

    Returns:
        bool: True if the peak lands within tolerance.
    """
    return delta_minutes <= tolerance_minutes
