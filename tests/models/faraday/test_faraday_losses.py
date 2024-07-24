import pytest
import torch
from torch.distributions.normal import Normal

from opensynth.models.faraday import losses


class TestFaradayLosses:

    mmd_tol = 0.001
    norm_dist_1 = Normal(0, 1)
    norm_dist_2 = Normal(25, 96)
    tensor1 = torch.tensor([1, 2, 3, 4, 5])
    tensor2 = torch.tensor([0, 0, 0, 0, 0])

    @pytest.mark.parametrize(
        "d1,d2,tol",
        [
            # Exact tensor should return quantile loss of 0
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
        got_mmd_loss = losses.MMDLoss(sample_1, sample_2)
        assert torch.round(got_mmd_loss, decimals=3) < tol

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
