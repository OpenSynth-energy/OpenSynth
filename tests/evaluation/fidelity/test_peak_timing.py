from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from opensynth.evaluation.fidelity import peak_timing


def _sinusoid_profile(peak_hour: int, intervals: int = 96) -> np.ndarray:
    hours = np.arange(intervals) / (intervals / 24)
    return 1.0 + np.cos((hours - peak_hour) / 24 * 2 * np.pi)


class TestSyntheticPeakHour:

    def test_known_peak_recovered(self):
        kwh = np.stack([_sinusoid_profile(18)] * 10)
        months = np.array([1, 1, 2, 2, 2, 12, 12, 1, 1, 2])
        hour = peak_timing.synthetic_peak_hour(kwh, months, [12, 1, 2])
        assert hour == 18.0

    def test_season_filter_applies(self):
        winter = np.stack([_sinusoid_profile(18)] * 5)
        summer = np.stack([_sinusoid_profile(14)] * 5)
        kwh = np.concatenate([winter, summer])
        months = np.array([1] * 5 + [7] * 5)
        assert peak_timing.synthetic_peak_hour(kwh, months, [7]) == 14.0

    def test_raises_on_empty_season(self):
        kwh = np.stack([_sinusoid_profile(18)] * 2)
        months = np.array([1, 1])
        with pytest.raises(ValueError):
            peak_timing.synthetic_peak_hour(kwh, months, [7])


class TestReferencePeakHour:

    def test_known_peak_recovered(self):
        start = datetime(2018, 1, 1)
        timestamps = [start + timedelta(hours=i) for i in range(24 * 60)]
        demand = [
            1000 + 200 * np.cos((t.hour - 17) / 24 * 2 * np.pi)
            for t in timestamps
        ]
        df = pl.DataFrame({"timestamp": timestamps, "demand_mwh": demand})
        assert peak_timing.reference_peak_hour(df, [1, 2]) == 17.0


class TestDeltaAndTolerance:

    def test_zero_delta_for_identical_peaks(self):
        assert peak_timing.peak_timing_delta_minutes(18, 18) == 0.0

    def test_delta_in_minutes(self):
        assert peak_timing.peak_timing_delta_minutes(18, 17) == 60.0

    def test_wraps_midnight(self):
        assert peak_timing.peak_timing_delta_minutes(23, 0) == 60.0

    def test_tolerance_check(self):
        assert peak_timing.check_peak_within_tolerance(60.0)
        assert not peak_timing.check_peak_within_tolerance(61.0)

    def test_downsample_rejects_bad_length(self):
        with pytest.raises(ValueError):
            peak_timing.downsample_to_hourly(np.ones(95))
