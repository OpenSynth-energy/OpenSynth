# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

"""Load-shape fidelity metrics.

Shape correlation and RMSE compare *normalised* mean daily profiles,
so they measure shape agreement independent of magnitude. The
energy-window share supports the EV-signature criterion: EV homes
should show a distinct evening charging uplift relative to non-EV
homes.
"""

from typing import Tuple

import numpy as np
from scipy.stats import pearsonr


def normalise_profile(profile: np.ndarray) -> np.ndarray:
    """
    Normalise a profile to unit total energy.

    Args:
        profile (np.ndarray): Daily profile.

    Returns:
        np.ndarray: Profile scaled to sum to 1.
    """
    profile = np.asarray(profile, dtype=float)
    total = profile.sum()
    if total == 0:
        raise ValueError("Cannot normalise an all-zero profile")
    return profile / total


def shape_correlation(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """
    Pearson correlation between two normalised daily profiles.

    Args:
        profile_a (np.ndarray): Daily profile.
        profile_b (np.ndarray): Daily profile of the same length.

    Returns:
        float: Correlation coefficient.
    """
    r, _ = pearsonr(normalise_profile(profile_a), normalise_profile(profile_b))
    return float(r)


def shape_rmse(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """
    RMSE between two normalised daily profiles.

    Args:
        profile_a (np.ndarray): Daily profile.
        profile_b (np.ndarray): Daily profile of the same length.

    Returns:
        float: Root mean squared error of the normalised profiles.
    """
    a = normalise_profile(profile_a)
    b = normalise_profile(profile_b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def energy_window_share(
    kwh: np.ndarray,
    window: Tuple[int, int] = (16, 23),
    periods_per_hour: int = 4,
) -> float:
    """
    Share of daily energy consumed within an hour window.

    Args:
        kwh (np.ndarray): Daily profiles [n_days, intervals].
        window (Tuple[int, int]): Start hour (inclusive) and end
            hour (exclusive). Defaults to the evening EV window.
        periods_per_hour (int): Intervals per hour. Defaults to 4.

    Returns:
        float: Fraction of total energy inside the window.
    """
    kwh = np.atleast_2d(np.asarray(kwh, dtype=float))
    start, end = window
    lo, hi = start * periods_per_hour, end * periods_per_hour
    total = kwh.sum()
    if total == 0:
        raise ValueError("Profiles contain no energy")
    return float(kwh[:, lo:hi].sum() / total)
