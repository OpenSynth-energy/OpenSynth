# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

"""Annual-consumption fidelity metric.

Compares per-home annual kWh of a synthetic dataset against the
RECS weighted regional distribution: mean percentage error,
two-sample Kolmogorov-Smirnov statistic and a decile comparison
table.
"""

from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp


@singledispatch
def annual_kwh_per_home(  # pragma: no cover
    df: Any, home_col: str = "home_id", kwh_col: str = "kwh"
) -> np.ndarray:
    """
    Per-home annual kWh totals from a long-format dataset.

    Args:
        df (DataFrame): Long rows with a home identifier and kwh.
        home_col (str): Home identifier column.
        kwh_col (str): Energy column.

    Returns:
        np.ndarray: One annual total per home.
    """
    raise NotImplementedError(f"Unsupported input type: {type(df)}")


@annual_kwh_per_home.register
def _(
    df: pl.DataFrame, home_col: str = "home_id", kwh_col: str = "kwh"
) -> np.ndarray:
    return (
        df.group_by(home_col)
        .agg(pl.col(kwh_col).sum())
        .sort(home_col)[kwh_col]
        .to_numpy()
    )


@annual_kwh_per_home.register
def _(
    df: pd.DataFrame, home_col: str = "home_id", kwh_col: str = "kwh"
) -> np.ndarray:
    return annual_kwh_per_home(pl.from_pandas(df), home_col, kwh_col)


def mean_percent_error(
    synthetic_kwh: np.ndarray, reference_mean: float
) -> float:
    """
    Signed percentage error of the synthetic mean annual kWh.

    Args:
        synthetic_kwh (np.ndarray): Per-home annual totals.
        reference_mean (float): Reference (RECS weighted) mean.

    Returns:
        float: Percentage error; positive means synthetic is high.
    """
    if reference_mean == 0:
        raise ValueError("Reference mean must be non-zero")
    return float(
        (np.mean(synthetic_kwh) - reference_mean) / reference_mean * 100.0
    )


def ks_statistic(
    synthetic_kwh: np.ndarray, reference_kwh: np.ndarray
) -> float:
    """
    Two-sample KS statistic between annual-kWh distributions.

    Args:
        synthetic_kwh (np.ndarray): Per-home annual totals.
        reference_kwh (np.ndarray): Reference per-home totals
            (e.g. RECS microdata KWH values).

    Returns:
        float: KS statistic (0 identical - 1 disjoint).
    """
    result = ks_2samp(synthetic_kwh, reference_kwh)
    return float(result.statistic)


def decile_table(
    synthetic_kwh: np.ndarray, reference_kwh: np.ndarray
) -> pl.DataFrame:
    """
    Decile-by-decile comparison of annual-kWh distributions.

    Args:
        synthetic_kwh (np.ndarray): Per-home annual totals.
        reference_kwh (np.ndarray): Reference per-home totals.

    Returns:
        pl.DataFrame: Columns decile, synthetic_kwh, reference_kwh,
        percent_error.
    """
    deciles = np.arange(0.1, 1.0, 0.1)
    synth_q = np.quantile(synthetic_kwh, deciles)
    ref_q = np.quantile(reference_kwh, deciles)
    return pl.DataFrame(
        {
            "decile": (deciles * 100).round().astype(int),
            "synthetic_kwh": synth_q,
            "reference_kwh": ref_q,
            "percent_error": (synth_q - ref_q) / ref_q * 100.0,
        }
    )
