import pytest
import torch

from opensynth.models.energydiff._diffusion_base import extract


@pytest.fixture(
    params=["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def device(request):
    return torch.device(request.param)


@pytest.fixture
def one_dim_tensor():
    """1D tensors like alpha_cumprod, (N, )"""
    return torch.randn(1000)


@pytest.fixture
def batched_timestep():
    return torch.randint(0, 1000, (16,))  # T=1000, B=16


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
    assert out.device == device
