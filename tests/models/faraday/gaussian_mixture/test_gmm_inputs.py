import pytest
import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.gaussian_mixture import (
    gmm_utils,
    initialise_gmm_params,
)
from opensynth.models.faraday.vae_model import FaradayVAE


class TestGMMDataPreparation:

    vae_module = FaradayVAE(
        class_dim=2, latent_dim=16, learning_rate=0.001, mse_weight=3
    )

    kwh = torch.rand(100, 48)
    features = {"feature_1": torch.rand(100), "feature_2": torch.rand(100)}
    weights = torch.randint(low=1, high=5, size=(100,))

    batch = TrainingData(kwh=kwh, features=features, weights=weights)

    unweighted_batch = TrainingData(kwh=kwh, features=features)

    sample_weights_column = "weights"

    def test_check_data_size(self):

        model_input = gmm_utils.prepare_data_for_training_step(
            self.batch, self.vae_module, self.sample_weights_column
        )

        assert (
            model_input.shape[0]
            == self.batch[self.sample_weights_column].sum()
        )
        assert model_input.shape[1] == self.vae_module.latent_dim + len(
            self.batch["features"].keys()
        )

    def test_check_weights(self):
        # Check weights > 1 have been incorporated correctly
        # If sample weight = 3, expect 3 rows of the same data in final tensor
        # Expect these rows to be shuffled into random positions in the final
        # tensor, not just repreated on consecutive rows.

        test_idx = torch.where(self.batch[self.sample_weights_column] > 1)[0][
            0
        ]
        encoded_batch = gmm_utils._encode_data(self.batch, self.vae_module)
        model_input = gmm_utils._expand_weights(
            encoded_batch, self.batch[self.sample_weights_column]
        )

        assert (
            torch.Tensor(
                [
                    torch.all(
                        model_input[i, :] == encoded_batch[test_idx, :]
                    ).item()
                    for i in range(model_input.size(0))
                ]
            ).sum()
            == self.batch[self.sample_weights_column][test_idx]
        )

        assert not torch.equal(
            torch.where(model_input == encoded_batch[test_idx, :])[0].unique(),
            torch.Tensor(
                [
                    test_idx + i
                    for i in range(
                        self.batch[self.sample_weights_column][test_idx]
                    )
                ]
            ),
        )

    def test_no_sample_weights(self):

        model_input = gmm_utils.prepare_data_for_training_step(
            self.batch, self.vae_module, sample_weights_column=None
        )

        assert model_input.size(0) == 100
        assert model_input.size(1) == self.vae_module.latent_dim + len(
            self.unweighted_batch["features"].keys()
        )

    def test_gmm_weights_sum_to_one_with_sample_weights(self):

        n_components = 2

        gmm_init_params = initialise_gmm_params(
            self.batch,
            n_components=n_components,
            vae_module=self.vae_module,
            sample_weights_column=self.sample_weights_column,
        )

        sum_weights = gmm_init_params["weights"].sum().numpy().round(2)

        assert sum_weights == 1.0

    def test_gmm_weights_sum_to_one_without_sample_weights(self):

        n_components = 2

        gmm_init_params = initialise_gmm_params(
            self.batch,
            n_components=n_components,
            vae_module=self.vae_module,
            sample_weights_column=None,
        )

        sum_weights = gmm_init_params["weights"].sum().numpy().round(2)

        assert sum_weights == 1.0

    @pytest.mark.xfail(raises=KeyError, strict=True)
    def test_wrong_weights_column(self):

        gmm_utils.prepare_data_for_training_step(
            self.batch, self.vae_module, "wrong_column"
        )
