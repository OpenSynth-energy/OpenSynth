import pytest
import torch
from torch.distributions.normal import Normal

from opensynth.models.faraday import losses


class TestFaradayLosses:

    mmd_tol = 0.001
    norm_dist_1 = Normal(0, 1)
    norm_dist_2 = Normal(25, 96)
    tensor1 = torch.tensor([1, 2, 3, 4, 5]).float()
    tensor2 = torch.tensor([0, 0, 0, 0, 0]).float()
    weights = torch.tensor([1, 1, 3, 1, 5])

    def test_expand_samples_1d_tensor(self):
        tensor1_expanded = losses._expand_samples(self.tensor1, self.weights)
        expected_tensor = torch.tensor(
            [1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5]
        ).float()
        assert torch.equal(
            tensor1_expanded.squeeze(),
            expected_tensor.squeeze(),
        )

    def test_expand_samples_2d_tensor(self):
        tensor1 = self.tensor1.reshape(len(self.tensor1), 1)
        tensor1_expanded = losses._expand_samples(tensor1, self.weights)
        expected_tensor_flat = torch.tensor(
            [1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 5]
        ).float()
        expected_tensor = expected_tensor_flat.reshape(
            len(expected_tensor_flat), 1
        )
        assert torch.equal(
            tensor1_expanded,
            expected_tensor,
        )

    @pytest.mark.parametrize(
        "d1,d2,tol",
        [
            # Exact tensor should return MMD loss of 0
            pytest.param(norm_dist_1, norm_dist_1, mmd_tol),
            pytest.param(
                norm_dist_1,
                norm_dist_2,
                mmd_tol,
                marks=pytest.mark.xfail(raises=AssertionError),
            ),
        ],
    )
    def test_mmd_loss(self, d1, d2, tol):
        sample_1 = d1.sample([5000, 2])
        sample_2 = d2.sample([5000, 2])
        got_mmd_loss = losses.mmd_loss(sample_1, sample_2)
        assert torch.round(got_mmd_loss, decimals=3) <= tol

    @pytest.mark.parametrize(
        "t1,t2,quantile,expected_value",
        [
            # Exact tensor should return quantile loss of 0
            pytest.param(tensor1, tensor1, 0.5, 0),
            pytest.param(tensor1, tensor2, 0.3, 2.1),
            pytest.param(tensor1, tensor2, 0.5, 1.5),
            pytest.param(tensor1, tensor2, 0.9, 0.3),
        ],
    )
    def test_quantile_loss(self, t1, t2, quantile, expected_value):
        got_loss = losses.quantile_loss(t1, t2, quantile)
        assert torch.round(got_loss, decimals=2) == expected_value

    @pytest.mark.parametrize(
        "t1,t2,quantile,expected_value",
        [
            # Exact tensor should return quantile loss of 0
            pytest.param(tensor1, tensor1, 0.5, 0),
            pytest.param(tensor1, tensor2, 0.5, 4.1),
        ],
    )
    def test_quantile_loss_with_weights(
        self, t1, t2, quantile, expected_value
    ):
        got_loss = losses.quantile_loss(t1, t2, quantile, self.weights)
        assert torch.round(got_loss, decimals=2) == expected_value

    @pytest.mark.parametrize(
        "t1,t2,expected_value",
        [
            # Exact tensor should return quantile loss of 0
            pytest.param(tensor1, tensor1, 0),
            pytest.param(tensor1, tensor2, 34.6),
        ],
    )
    def test_mse_loss(self, t1, t2, expected_value):
        got_loss = losses.mse_loss(t1, t2, self.weights)
        assert got_loss == expected_value
