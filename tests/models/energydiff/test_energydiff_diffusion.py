from typing import Tuple

import pytest
import torch
from torch import Tensor

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.energydiff.diffusion import (
    BetaScheduleType,
    GaussianDiffusion1D,
    LossType,
    ModelMeanType,
    ModelVarianceType,
    PLDiffusion1D,
)
from opensynth.models.energydiff.model import DenoisingTransformer


@pytest.fixture(
    params=["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else [])
)
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=list(ModelMeanType))
def model_mean_type(request) -> ModelMeanType:
    return request.param


@pytest.fixture(
    params=[
        ModelVarianceType.FIXED_SMALL,
    ]
)
def model_variance_type(request) -> ModelVarianceType:
    return request.param


@pytest.fixture(
    params=[
        LossType.MSE,
        LossType.RESCALED_MSE,
    ]
)  # KL-related not implemented.
def loss_type(request) -> LossType:
    return request.param


@pytest.fixture(params=[BetaScheduleType.LINEAR, BetaScheduleType.COSINE])
def beta_schedule_type(request) -> BetaScheduleType:
    return request.param


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def profile_seq_len() -> int:
    return 180


@pytest.fixture
def profile_channel() -> int:
    return 8


@pytest.fixture
def num_timestep() -> int:
    return 1000


def make_input_profile(
    batch_size, profile_seq_len, profile_channel, device
) -> Tensor:
    return torch.randn(batch_size, profile_seq_len, profile_channel).to(device)


@pytest.fixture()
def denoising_model_hparams(profile_channel) -> dict:
    return {
        "dim_base": 48,
        "dim_in": profile_channel,
        "num_attn_head": 12,
        "num_decoder_layer": 6,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "learn_variance": False,
        "disable_init_proj": False,
    }


@pytest.fixture()
def denoising_model(device, denoising_model_hparams) -> DenoisingTransformer:
    hparams = denoising_model_hparams
    return DenoisingTransformer(**hparams).to(device)


@pytest.mark.usefixtures("profile_seq_len")
@pytest.mark.usefixtures("profile_channel")
@pytest.mark.usefixtures("batch_size")
@pytest.mark.usefixtures("device")
class Test_GaussianDiffusion1D:
    @pytest.fixture
    def diffusion_instance(
        self,
        denoising_model,
        num_timestep,
        model_mean_type,
        model_variance_type,
        loss_type,
        beta_schedule_type,
    ) -> GaussianDiffusion1D:
        return GaussianDiffusion1D(
            base_model=denoising_model,
            num_timestep=num_timestep,
            model_mean_type=model_mean_type,
            model_variance_type=model_variance_type,
            loss_type=loss_type,
            beta_schedule_type=beta_schedule_type,
        )

    @pytest.fixture
    def input_profile(
        self, batch_size, profile_seq_len, profile_channel, device
    ) -> Tensor:
        return make_input_profile(
            batch_size, profile_seq_len, profile_channel, device
        )

    def test_init(self, diffusion_instance):
        pass

    @pytest.fixture(
        params=[
            pytest.param(True, id="given_noise"),
            pytest.param(False, id="no_given_noise"),
        ]
    )
    def given_noise(self, request) -> bool:
        return request.param

    def test_train_losses(
        self, diffusion_instance, input_profile, given_noise, device
    ):
        batched_time = torch.randint(
            0, diffusion_instance.num_timestep, (input_profile.shape[0],)
        ).to(device)
        if given_noise:
            noise = torch.randn_like(input_profile)
        else:
            noise = None
        losses = diffusion_instance.train_losses(
            input_profile, batched_time, noise
        )
        for value in losses.values():
            assert isinstance(value, torch.Tensor)

    def test_forward(
        self,
        diffusion_instance,
        input_profile,
        given_noise,
    ):
        if given_noise:
            noise = torch.randn_like(input_profile)
        else:
            noise = None
        loss_terms = diffusion_instance.forward(input_profile, noise)
        for value in loss_terms.values():
            assert isinstance(value, torch.Tensor)
            assert value.device == input_profile.device

    def test_sample(
        self,
        diffusion_instance,
        profile_seq_len,
        profile_channel,
    ):
        num_sample = 5
        sample = diffusion_instance.dpm_solver_sample(
            total_num_sample=num_sample,
            batch_size=4,
            step=15,  # sampling step
            shape=(profile_seq_len, profile_channel),
        )
        assert sample.shape == (num_sample, profile_seq_len, profile_channel)

    def test_backward(
        self,
        diffusion_instance,
        input_profile,
        given_noise,
    ):
        if given_noise:
            noise = torch.randn_like(input_profile)
        else:
            noise = None
        loss_terms = diffusion_instance.forward(input_profile, noise)
        loss = loss_terms["loss"]
        loss.backward()
        for param in diffusion_instance.model.parameters():
            assert param.grad is not None


@pytest.fixture
def fake_batch() -> TrainingData:
    return TrainingData(kwh=torch.randn(32, 48), features={})


class Test_PLDiffusion1D:
    @pytest.fixture
    def pl_module(
        self,
        denoising_model_hparams,
        num_timestep,
        model_mean_type,
        model_variance_type,
        loss_type,
        beta_schedule_type,
    ) -> PLDiffusion1D:
        return PLDiffusion1D(
            dim_base=denoising_model_hparams["dim_base"],
            dim_in=1,  # for LCL-like (batch, 48, 1)
            num_attn_head=denoising_model_hparams["num_attn_head"],
            num_decoder_layer=denoising_model_hparams["num_decoder_layer"],
            dim_feedforward=denoising_model_hparams["dim_feedforward"],
            dropout=denoising_model_hparams["dropout"],
            learn_variance=denoising_model_hparams["learn_variance"],
            num_timestep=num_timestep,
            model_mean_type=model_mean_type,
            model_variance_type=model_variance_type,
            loss_type=loss_type,
            beta_schedule_type=beta_schedule_type,
            lr=1e-4,
            ema_update_every=3,
            ema_decay=0.999,
            disable_init_proj=denoising_model_hparams["disable_init_proj"],
        )

    @pytest.fixture
    def optimizer_instance(
        self,
        pl_module,
    ) -> torch.optim.Optimizer:
        return pl_module.configure_optimizers()

    def test_init(self, pl_module):
        pass

    def test_optimizer_init(self, optimizer_instance):
        pass

    @pytest.fixture
    def processed_batch(self, pl_module, fake_batch) -> Tuple[Tensor, int]:
        batch_idx = 33
        processed_batch = pl_module.on_before_batch_transfer(
            fake_batch, batch_idx
        )
        return processed_batch, batch_idx

    @pytest.fixture(scope="function")
    def one_step_loss_terms(
        self,
        pl_module,
        processed_batch,
    ) -> Tensor:
        loss = pl_module.training_step(*processed_batch)

        return loss

    def test_training_step(self, one_step_loss_terms):
        loss = one_step_loss_terms
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_on_train_batch_end(
        self, pl_module, one_step_loss_terms, processed_batch
    ):
        out = pl_module.on_train_batch_end(
            one_step_loss_terms, *processed_batch
        )
        assert out is None

    def test_validation_step(self, pl_module, processed_batch):
        out = pl_module.validation_step(*processed_batch)
        assert out is None

    def test_optimizer_step(
        self, pl_module, optimizer_instance, one_step_loss_terms
    ):
        loss = one_step_loss_terms
        optimizer_instance.zero_grad()
        loss.backward()
        optimizer_instance.step()
        for param in pl_module.diffusion_model.parameters():
            assert param.grad is not None


if __name__ == "__main__":
    B, L, C = 32, 180, 8
    hparams = {
        "dim_base": 48,
        "dim_in": C,
        "num_attn_head": 12,
        "num_decoder_layer": 6,
        "dim_feedforward": 256,
        "dropout": 0.3,
        "learn_variance": False,
        "disable_init_proj": False,
    }
    nn_model = DenoisingTransformer(**hparams).to("cpu")
    df = GaussianDiffusion1D(
        nn_model,
        1000,
        ModelMeanType.V,
        ModelVarianceType.FIXED_SMALL,
        LossType.MSE,
        BetaScheduleType.LINEAR,
    )
    sample = df.dpm_solver_sample(
        total_num_sample=5,
        batch_size=5,
        step=15,  # sampling step
        shape=(L, C),
    )
