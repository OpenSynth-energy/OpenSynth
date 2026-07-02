import math

import pytest
import torch

from opensynth.datasets.new_england.faraday_ne import NewEnglandFaradayModel
from opensynth.models.faraday.gaussian_mixture.gmm_model import (
    GaussianMixtureModel,
)
from opensynth.models.faraday.vae_model import FaradayVAE

LATENT_DIM = 2
N_LABELS = 1  # single conditioning feature: temp_bin


def _build_model(means, covariances, weights):
    vae = FaradayVAE(class_dim=N_LABELS, latent_dim=LATENT_DIM, input_dim=4)
    vae.feature_list = ["temp_bin"]
    gmm = GaussianMixtureModel(
        num_components=len(weights),
        num_features=LATENT_DIM + N_LABELS,
    )
    gmm.means.data = torch.tensor(means, dtype=torch.float32)
    gmm.covariances.data = torch.tensor(covariances, dtype=torch.float32)
    gmm.weights.data = torch.tensor(weights, dtype=torch.float32)
    model = NewEnglandFaradayModel(vae_module=vae, n_components=len(weights))
    model.gmm_module = gmm
    return model


def _two_component_model():
    # Independent latent/label blocks; components separated in both
    # latent space and label space (y=0 vs y=1)
    eye3 = torch.eye(3).tolist()
    return _build_model(
        means=[[0.0, 0.0, 0.0], [5.0, 5.0, 1.0]],
        covariances=[eye3, eye3],
        weights=[0.5, 0.5],
    )


class TestConditionalMixture:

    def test_component_reweighting_matches_analytic(self):
        model = _two_component_model()
        probs, _, _ = model._conditional_mixture(
            torch.tensor([1.0], dtype=torch.float64)
        )
        # With equal priors and unit variances the posterior odds are
        # N(1;1,1) : N(1;0,1) = 1 : exp(-0.5)
        expected_p1 = 1.0 / (1.0 + math.exp(-0.5))
        assert probs[1].item() == pytest.approx(expected_p1, abs=1e-4)

    def test_conditional_mean_uses_cross_covariance(self):
        # One component, latent dim 0 correlated with the label:
        # E[z0 | y] = mu_z0 + cov_zy / var_y * (y - mu_y)
        cov = [
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.0],
            [0.8, 0.0, 1.0],
        ]
        model = _build_model(
            means=[[0.0, 0.0, 0.0]], covariances=[cov], weights=[1.0]
        )
        _, cond_means, _ = model._conditional_mixture(
            torch.tensor([2.0], dtype=torch.float64)
        )
        assert cond_means[0][0].item() == pytest.approx(1.6, abs=1e-3)
        assert cond_means[0][1].item() == pytest.approx(0.0, abs=1e-3)

    def test_conditional_covariance_shrinks(self):
        cov = [
            [1.0, 0.0, 0.8],
            [0.0, 1.0, 0.0],
            [0.8, 0.0, 1.0],
        ]
        model = _build_model(
            means=[[0.0, 0.0, 0.0]], covariances=[cov], weights=[1.0]
        )
        _, _, cond_chol = model._conditional_mixture(
            torch.tensor([0.0], dtype=torch.float64)
        )
        cond_var_z0 = (cond_chol[0] @ cond_chol[0].T)[0, 0].item()
        # 1 - 0.8^2 / 1 = 0.36
        assert cond_var_z0 == pytest.approx(0.36, abs=1e-2)


class TestSampleGmmConditional:

    def test_returns_requested_labels_and_shape(self):
        model = _two_component_model()
        torch.manual_seed(0)
        out = model.sample_gmm_conditional({"temp_bin": 1}, n_samples=50)
        assert out["kwh"].shape == (50, 4)
        assert torch.isfinite(out["kwh"]).all()
        assert (out["features"]["temp_bin"] == 1.0).all()
        assert out["features"]["temp_bin"].shape == (50, 1)

    def test_rejects_unknown_labels(self):
        model = _two_component_model()
        with pytest.raises(ValueError):
            model.sample_gmm_conditional({"month": 1}, n_samples=5)

    def test_conditioning_shifts_latent_selection(self):
        # y=1 should overwhelmingly select component 1 when the
        # components are far apart in label space
        eye3 = torch.eye(3).tolist()
        model = _build_model(
            means=[[0.0, 0.0, 0.0], [5.0, 5.0, 10.0]],
            covariances=[eye3, eye3],
            weights=[0.5, 0.5],
        )
        probs, _, _ = model._conditional_mixture(
            torch.tensor([10.0], dtype=torch.float64)
        )
        assert probs[1].item() > 0.999
