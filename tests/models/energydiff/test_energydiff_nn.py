import pytest
import torch
from torch import Tensor

from opensynth.models.energydiff.model import (
    DecoderTransformer,
    DenoisingTransformer,
    SinusoidalPosEmb,
)


@pytest.fixture(
    params=["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else [])
)
def device(request):
    return torch.device(request.param)


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def profile_seq_len() -> int:
    return 180


@pytest.fixture
def profile_channel() -> int:
    return 8


@pytest.mark.usefixtures("device")
class Test_SinusoidalPosEmb:
    @pytest.fixture
    def dim(self) -> int:
        return 64

    @pytest.fixture
    def module(self, dim, device):
        return SinusoidalPosEmb(dim).to(device)

    def test_init(self, module):
        pass

    @pytest.fixture
    def position(self, batch_size, device) -> Tensor:
        pos = torch.randint(0, 10000, (batch_size,)).to(device)
        return pos

    def test_forward(self, module, position):
        result = module(position)
        assert result.shape == (position.shape[0], module.dim)
        assert result.device == position.device


def make_input_profile(
    batch_size, profile_seq_len, profile_channel, device
) -> Tensor:
    return torch.randn(batch_size, profile_seq_len, profile_channel).to(device)


def make_encoded_profile(
    batch_size, profile_seq_len, dim_base, device
) -> Tensor:
    return torch.randn(batch_size, profile_seq_len, dim_base).to(device)


def make_encoded_condition(batch_size, dim_base, device) -> Tensor:
    return torch.randn(batch_size, 1, dim_base).to(device)


@pytest.fixture(
    params=[
        "unconditional",
        "conditional",
    ]
)
def decoder_hparams(request) -> dict:
    hparams = {
        "dim_base": 42,
        "num_attn_head": 6,
        "num_layer": 3,
        "dim_feedforward": 256,
        "dropout": 0.2,
        "conditioning": False,
    }
    if request.param == "conditional":
        hparams["conditioning"] = True
    return hparams


@pytest.mark.usefixtures("batch_size")
@pytest.mark.usefixtures("device")
class Test_DecoderTransformer:
    @pytest.fixture
    def module(self, decoder_hparams, device):
        return DecoderTransformer(**decoder_hparams).to(device)

    def test_init(self, module):
        pass

    @pytest.fixture
    def encoded_profile(self, batch_size, profile_seq_len, module, device):
        return make_encoded_profile(
            batch_size, profile_seq_len, module.dim_base, device
        )

    @pytest.fixture(params=["cond", "no-cond"])
    def encoded_condition(self, request, batch_size, module, device):
        if request.param == "given":
            return make_encoded_condition(batch_size, module.dim_base, device)
        else:
            return None

    def test_forward(self, module, encoded_profile, encoded_condition):
        result = module(encoded_profile, encoded_condition)
        assert result.shape == encoded_profile.shape
        assert result.device == encoded_profile.device


@pytest.fixture(
    params=[
        "disable_init_proj",
        "enable_init_proj",
    ]
)
def denoising_transformer_hparams(profile_seq_len, profile_channel) -> dict:
    hparam = {
        "dim_base": 48,
        "dim_in": profile_channel,
        "num_attn_head": 12,
        "num_decoder_layer": 6,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "learn_variance": False,
        "disable_init_proj": False,
    }
    if hparam == "disable_init_proj":
        hparam["disable_init_proj"] = True
    return hparam


@pytest.mark.usefixtures("batch_size")
@pytest.mark.usefixtures("device")
class Test_DenoisingTransformer:
    @pytest.fixture
    def module(self, denoising_transformer_hparams, device):
        return DenoisingTransformer(**denoising_transformer_hparams).to(device)

    def test_init(self, module):
        pass

    @pytest.fixture
    def input_profile(
        self, batch_size, profile_seq_len, profile_channel, device
    ):
        return make_input_profile(
            batch_size, profile_seq_len, profile_channel, device
        )

    @pytest.fixture
    def input_time(self, batch_size, device):
        return torch.randint(0, 1234, (batch_size,)).to(device)

    def test_forward(self, module, input_profile, input_time):
        out = module(input_profile, input_time)
        assert out.device == input_profile.device
        if not module.learn_variance:
            assert out.shape == input_profile.shape
        else:
            assert out.shape[:-1] == input_profile.shape[:-1]
            assert out.shape[-1] == 2 * input_profile.shape[-1]
