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

    def test_mmd_loss_weighted_and_expanded_same_values(self):

        N = 1000
        sample_1 = self.norm_dist_1.sample([N, 2])
        sample_2 = self.norm_dist_2.sample([N, 2])
        weights = torch.randint(low=1, high=3, size=(N, 1))
        assert weights.sum() > N

        got_mmd_loss = losses.mmd_loss(sample_1, sample_2, weights)
        got_mmd_loss = torch.round(got_mmd_loss, decimals=3)

        # Expanded samples
        sample_1_expanded = losses._expand_samples(sample_1, weights)
        sample_2_expanded = losses._expand_samples(sample_2, weights)

        # Check that expanded samples are larger than original samples!
        assert len(sample_1_expanded) == weights.sum()
        assert len(sample_2_expanded) == weights.sum()

        expected_mmd_loss = losses.mmd_loss(
            sample_1_expanded, sample_2_expanded
        )
        expected_mmd_loss = torch.round(expected_mmd_loss, decimals=3)

        assert got_mmd_loss == expected_mmd_loss

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
            pytest.param(tensor1, tensor2, 0.5, 1.86),
        ],
    )
    def test_quantile_loss_with_weights(
        self, t1, t2, quantile, expected_value
    ):
        got_loss = losses.quantile_loss(t1, t2, quantile, self.weights)
        assert torch.round(got_loss, decimals=2) == expected_value

    def test_quantile_loss_weighted_and_expanded_same_values(self):
        # Test supplying weights gives the same
        # results as manually expanding the tensors and
        # calculating the quantile loss
        weighted_quantile_loss = losses.quantile_loss(
            self.tensor1, self.tensor2, 0.5, self.weights
        )
        tensor1_expanded = losses._expand_samples(self.tensor1, self.weights)
        tensor2_expanded = losses._expand_samples(self.tensor2, self.weights)
        unweighted_quantile_loss = losses.quantile_loss(
            tensor1_expanded, tensor2_expanded, 0.5, None
        )

        assert torch.round(
            unweighted_quantile_loss, decimals=2
        ) == torch.round(weighted_quantile_loss, decimals=2)

    @pytest.mark.parametrize(
        "t1,t2,expected_value",
        [
            # Exact tensor should return quantile loss of 0
            pytest.param(tensor1, tensor1, 0),
            pytest.param(tensor1, tensor2, 15.73),
        ],
    )
    def test_mse_loss(self, t1, t2, expected_value):
        got_loss = losses.mse_loss(t1, t2, self.weights)
        assert torch.round(got_loss, decimals=2) == expected_value

    def test_mse_loss_weighted_and_expanded_same_values(self):
        tensor1_expanded = losses._expand_samples(self.tensor1, self.weights)
        tensor2_expanded = losses._expand_samples(self.tensor2, self.weights)
        weighted_mse_loss = losses.mse_loss(
            self.tensor1, self.tensor2, self.weights
        )
        unweighted_mse_loss = losses.mse_loss(
            tensor1_expanded, tensor2_expanded, None
        )
        assert torch.round(weighted_mse_loss, decimals=2) == torch.round(
            unweighted_mse_loss, decimals=2
        )
