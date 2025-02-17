import numpy as np
import pytest
import torch

from opensynth.models.energydiff._diffusion_base import (
    BetaScheduleType,
    ModelMeanType,
    get_beta_schedule,
    get_loss_weight,
)


@pytest.fixture
def alpha_cumprod() -> torch.Tensor:
    alpha_cumprod = np.random.rand(100)
    return torch.from_numpy(alpha_cumprod).float()


@pytest.fixture(params=list(ModelMeanType))
def model_mean_type(request) -> ModelMeanType:
    return request.param


@pytest.fixture(params=[BetaScheduleType.LINEAR, BetaScheduleType.COSINE])
def beta_schedule_type(request) -> BetaScheduleType:
    """only allows linear and cosine. discrete is only used for DPMSolver"""
    return request.param


def test_get_loss_weight(alpha_cumprod, model_mean_type):
    loss_weight = get_loss_weight(alpha_cumprod, model_mean_type)
    assert loss_weight.shape == alpha_cumprod.shape
    assert isinstance(loss_weight, torch.Tensor)


def test_get_beta_schedule(beta_schedule_type):
    num_timestep = 100
    beta_schedule = get_beta_schedule(beta_schedule_type, num_timestep)
    assert beta_schedule.shape == (num_timestep,)
    assert isinstance(beta_schedule, torch.Tensor)
    assert torch.all(beta_schedule >= 0)
    assert torch.all(beta_schedule <= 0.9999)
    if beta_schedule_type == BetaScheduleType.LINEAR:
        assert torch.all(beta_schedule[1:] - beta_schedule[:-1] >= 0)
    elif beta_schedule_type == BetaScheduleType.COSINE:
        assert torch.all(beta_schedule[1:] - beta_schedule[:-1] >= 0)
    else:
        raise ValueError("test: beta_schedule must be one of linear, cosine")
    # boundary check
    assert torch.isclose(
        beta_schedule[0], torch.zeros_like(beta_schedule)[0], atol=1e-3
    ), f"beta_schedule[0] {beta_schedule[0]} should be 0"
    # no need to check beta_schedule[-1], as it varies. can check alpha
