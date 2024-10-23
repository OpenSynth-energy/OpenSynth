import pytest

from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.gaussian_mixture import GaussianMixtureModel
from opensynth.models.faraday.gaussian_mixture.gmm_lightning import (
    GaussianMixtureInitLightningModule,
    GaussianMixtureLightningModule,
)


class TestGmmLightningSampleWeightInvocation:

    vae_module = FaradayVAE(
        class_dim=2, latent_dim=16, learning_rate=0.001, mse_weight=3
    )

    gmm_module = GaussianMixtureModel(2, 2)

    def test_gmm_init_module_sample_weights_true(self):
        init_module = GaussianMixtureInitLightningModule(
            self.gmm_module,
            self.vae_module,
            num_components=2,
            num_features=2,
            train_sample_weights=True,
        )
        assert init_module

    @pytest.mark.xfail(raises=TypeError, strict=True)
    def test_gmm_init_module_sample_weights_missing(self):
        init_module = GaussianMixtureInitLightningModule(
            self.gmm_module,
            self.vae_module,
            num_components=2,
            num_features=2,
        )
        assert init_module

    def test_gmm_lightning_module_sample_weights_true(self):
        gmm_module = GaussianMixtureLightningModule(
            self.gmm_module,  # init_module.model,
            self.vae_module,
            num_components=2,
            num_features=2,
            train_sample_weights=True,
        )
        assert gmm_module

    @pytest.mark.xfail(raises=TypeError, strict=True)
    def test_gmm_lightning_module_sample_weights_missing(self):
        gmm_module = GaussianMixtureLightningModule(
            self.gmm_module,  # init_module.model,
            self.vae_module,
            num_components=2,
            num_features=2,
        )
        assert gmm_module
