import numpy as np
import pytest
import torch

from opensynth.models.energydiff._diffusion_base import (
    BetaScheduleType,
    ModelMeanType,
    extract,
    get_beta_schedule,
    get_loss_weight,
)


@pytest.fixture
def num_timestep() -> int:
    return 100


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


@pytest.fixture(
    params=["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else [])
)
def device(request):
    return torch.device(request.param)


def test_get_loss_weight(alpha_cumprod, model_mean_type):
    loss_weight = get_loss_weight(alpha_cumprod, model_mean_type)
    assert loss_weight.shape == alpha_cumprod.shape
    assert isinstance(loss_weight, torch.Tensor)


def test_get_beta_schedule(num_timestep, beta_schedule_type):
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


@pytest.fixture
def one_dim_tensor():
    """1D tensors like alpha_cumprod, (N, )"""
    return torch.randn(1000)


@pytest.fixture
def batched_timestep(num_timestep):
    return torch.randint(0, num_timestep, (16,))  # T=1000, B=16


@pytest.fixture
def profile_tensor():
    """shape x: (batch, sequence, channel)"""
    return torch.randn(16, 180, 8)


def test_extract(one_dim_tensor, batched_timestep, profile_tensor, device):
    one_dim_tensor = one_dim_tensor.to(device)
    batched_timestep = batched_timestep.to(device)
    profile_tensor = profile_tensor.to(device)

    out = extract(one_dim_tensor, batched_timestep, profile_tensor.shape)
    assert out.shape == profile_tensor.shape
    assert out.device == batched_timestep.device


def test_extract_beta(
    num_timestep, beta_schedule_type, batched_timestep, profile_tensor, device
):
    beta_schedule = get_beta_schedule(beta_schedule_type, num_timestep)
    batched_timestep = batched_timestep.to(device)
    profile_tensor = profile_tensor.to(device)
    extracted = extract(beta_schedule, batched_timestep, profile_tensor.shape)
    assert extracted.shape == profile_tensor.shape
    assert extracted.device == batched_timestep.device
