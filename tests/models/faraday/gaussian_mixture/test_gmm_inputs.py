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

    def test_check_data_size(self):

        model_input = gmm_utils.encode_data(
            self.batch,
            self.vae_module,
        )
        model_input = gmm_utils.expand_weights(
            model_input, self.batch["weights"]
        )

        assert model_input.shape[0] == self.batch["weights"].sum()
        assert model_input.shape[1] == self.vae_module.latent_dim + len(
            self.batch["features"].keys()
        )

    def test_check_weights(self):
        # Check weights > 1 have been incorporated correctly
        # If sample weight = 3, expect 3 rows of the same data in final tensor
        # Expect these rows to be shuffled into random positions in the final
        # tensor, not just repreated on consecutive rows.

        test_idx = torch.where(self.batch["weights"] > 1)[0][0]
        encoded_batch = gmm_utils.encode_data(self.batch, self.vae_module)
        model_input = gmm_utils.expand_weights(
            encoded_batch, self.batch["weights"]
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
            == self.batch["weights"][test_idx]
        )

        assert not torch.equal(
            torch.where(model_input == encoded_batch[test_idx, :])[0].unique(),
            torch.Tensor(
                [test_idx + i for i in range(self.batch["weights"][test_idx])]
            ),
        )

    def test_no_weights(self):
        model_input = gmm_utils.encode_data(self.batch, self.vae_module)

        assert model_input.size(0) == 100
        assert model_input.size(1) == self.vae_module.latent_dim + len(
            self.unweighted_batch["features"].keys()
        )

    def test_weights_sum_to_one(self):

        n_components = 2

        gmm_init_params = initialise_gmm_params(
            self.batch,
            n_components=n_components,
            vae_module=self.vae_module,
        )

        sum_weights = gmm_init_params["weights"].sum().numpy().round(2)

        assert sum_weights == 1.0
