import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    encode_data_for_gmm,
    expand_weights,
    prepare_data_for_model,
)


class TestGMMDataPreparation:

    vae_module = FaradayVAE(
        class_dim=2, latent_dim=16, learning_rate=0.001, mse_weight=3
    )

    batch = TrainingData(
        kwh=torch.rand(100, 48),
        features={"feature_1": torch.rand(100), "feature_2": torch.rand(100)},
        weights=torch.randint(low=1, high=5, size=(100,)),
    )

    def check_data_size(self):

        model_input = prepare_data_for_model(self.vae_module, self.batch)

        assert model_input.shape[0] == self.batch["weights"].sum()
        assert model_input.shape[1] == self.vae_module.latent_dim + len(
            self.batch["features"].keys()
        )

    def check_weights(self):
        # Check weights > 1 have been incorporated correctly
        # If sample weight = 3, expect 3 rows of the same data in final tensor
        # Expect these rows to be shuffled into random positions in the final
        # tensor, not just repreated on consecutive rows.

        test_idx = torch.where(self.batch["weights"] > 1)[0][0]
        encoded_batch = encode_data_for_gmm(self.batch, self.vae_module)
        model_input = expand_weights(encoded_batch, self.batch["weights"])

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

        batch_no_weight = TrainingData(
            kwh=torch.rand(100, 48),
            features={
                "feature_1": torch.rand(100),
                "feature_2": torch.rand(100),
            },
        )
        model_input = prepare_data_for_model(self.vae_module, batch_no_weight)

        assert model_input.size(0) == 100
        assert model_input.size(1) == self.vae_module.latent_dim + len(
            batch_no_weight["features"].keys()
        )
