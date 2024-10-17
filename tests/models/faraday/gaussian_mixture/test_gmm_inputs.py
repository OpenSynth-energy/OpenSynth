import pytest
import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    encode_data_for_gmm,
    expand_weights,
    prepare_data_for_model,
)


class TrainingDataWithWeights(TrainingData):
    weights: torch.Tensor


class TestGMMDataPreparation:

    vae_module = FaradayVAE(
        class_dim=2, latent_dim=16, learning_rate=0.001, mse_weight=3
    )

    kwh = torch.rand(100, 48)
    features = {"feature_1": torch.rand(100), "feature_2": torch.rand(100)}
    weights = torch.randint(low=1, high=5, size=(100,))

    weighted_batch = TrainingDataWithWeights(
        kwh=kwh, features=features, weights=weights
    )
    unweighted_batch = TrainingData(kwh=kwh, features=features)

    def test_check_data_size(self):

        model_input = prepare_data_for_model(
            vae_module=self.vae_module,
            data=self.weighted_batch,
            sample_weight_col="weights",
        )

        assert model_input.shape[0] == self.weighted_batch["weights"].sum()
        assert model_input.shape[1] == self.vae_module.latent_dim + len(
            self.weighted_batch["features"].keys()
        )

    def test_check_weights(self):
        # Check weights > 1 have been incorporated correctly
        # If sample weight = 3, expect 3 rows of the same data in final tensor
        # Expect these rows to be shuffled into random positions in the final
        # tensor, not just repreated on consecutive rows.

        test_idx = torch.where(self.weighted_batch["weights"] > 1)[0][0]
        encoded_batch = encode_data_for_gmm(
            self.weighted_batch, self.vae_module
        )
        model_input = expand_weights(
            encoded_batch, self.weighted_batch["weights"]
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
            == self.weighted_batch["weights"][test_idx]
        )

        assert not torch.equal(
            torch.where(model_input == encoded_batch[test_idx, :])[0].unique(),
            torch.Tensor(
                [
                    test_idx + i
                    for i in range(self.weighted_batch["weights"][test_idx])
                ]
            ),
        )

    @pytest.mark.parametrize(
        "batch_data, sample_weight_col",
        [
            # Exact tensor should return quantile loss of 0
            pytest.param(weighted_batch, "weights"),
            pytest.param(unweighted_batch, ""),
            pytest.param(
                weighted_batch,
                True,  # Not string
                marks=pytest.mark.xfail(raises=TypeError, strict=True),
            ),
            pytest.param(
                weighted_batch,
                "different_column",  # Column not found
                marks=pytest.mark.xfail(raises=KeyError, strict=True),
            ),
        ],
    )
    def test_expect_weight_if_train_sample_weights_provided(
        self, batch_data, sample_weight_col
    ):
        model_input = prepare_data_for_model(
            self.vae_module, batch_data, sample_weight_col
        )
        assert len(model_input) > 0

    def test_no_weights(self):
        model_input = prepare_data_for_model(
            self.vae_module, self.unweighted_batch
        )

        assert model_input.size(0) == 100
        assert model_input.size(1) == self.vae_module.latent_dim + len(
            self.unweighted_batch["features"].keys()
        )

    def test_bespoke_weight_col(self):

        class BespokeData(TrainingData):
            bespoke_weights: torch.Tensor

        bespoke_batch = BespokeData(
            kwh=self.kwh, features=self.features, bespoke_weights=self.weights
        )

        model_inputs = prepare_data_for_model(
            vae_module=self.vae_module,
            data=bespoke_batch,
            sample_weight_col="bespoke_weights",
        )
        assert len(model_inputs) > 0
