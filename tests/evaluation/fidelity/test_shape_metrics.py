import numpy as np
import pytest

from opensynth.evaluation.fidelity import shape_metrics


class TestShapeCorrelation:

    def test_identical_profiles_correlate_perfectly(self):
        profile = np.sin(np.linspace(0, np.pi, 24)) + 1
        assert shape_metrics.shape_correlation(
            profile, profile
        ) == pytest.approx(1.0)

    def test_scaling_does_not_change_correlation(self):
        profile = np.sin(np.linspace(0, np.pi, 24)) + 1
        assert shape_metrics.shape_correlation(
            profile, profile * 7.5
        ) == pytest.approx(1.0)

    def test_inverted_profile_anticorrelates(self):
        profile = np.sin(np.linspace(0, np.pi, 24)) + 1
        inverted = profile.max() + profile.min() - profile
        assert shape_metrics.shape_correlation(
            profile, inverted
        ) == pytest.approx(-1.0)


class TestShapeRmse:

    def test_identical_profiles_have_zero_rmse(self):
        profile = np.sin(np.linspace(0, np.pi, 24)) + 1
        assert shape_metrics.shape_rmse(profile, profile) == 0.0

    def test_scaling_does_not_change_rmse(self):
        profile = np.sin(np.linspace(0, np.pi, 24)) + 1
        assert shape_metrics.shape_rmse(
            profile, profile * 3.0
        ) == pytest.approx(0.0)

    def test_zero_profile_rejected(self):
        with pytest.raises(ValueError):
            shape_metrics.shape_rmse(np.zeros(24), np.ones(24))

    def test_negative_net_load_profile_rejected(self):
        # Normalising by a negative total silently inverts the shape
        exporting_home = np.full(24, -0.5)
        with pytest.raises(ValueError):
            shape_metrics.normalise_profile(exporting_home)


class TestEnergyWindowShare:

    def test_all_energy_in_window(self):
        kwh = np.zeros((3, 96))
        kwh[:, 16 * 4 : 23 * 4] = 1.0
        assert shape_metrics.energy_window_share(kwh) == pytest.approx(1.0)

    def test_uniform_energy_share_matches_window_fraction(self):
        kwh = np.ones((5, 96))
        assert shape_metrics.energy_window_share(kwh) == pytest.approx(7 / 24)

    def test_ev_uplift_detectable(self):
        base = np.ones((10, 96)) * 0.1
        ev = base.copy()
        ev[:, 19 * 4 : 21 * 4] += 1.8  # evening charging block
        assert shape_metrics.energy_window_share(
            ev
        ) > shape_metrics.energy_window_share(base)

    def test_negative_net_energy_rejected(self):
        with pytest.raises(ValueError):
            shape_metrics.energy_window_share(np.full((2, 96), -0.1))
