# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
#
# The EnergyDiff model is made available via Nan Lin and Pedro P. Vergara
# from the Delft University of Technology. Nan Lin and Pedro P. Vergara are
# funded via the ALIGN4Energy Project (with project number NWA.1389.20.251) of
# the research programme NWA ORC 2020 which is (partly) financed by the Dutch
# Research Council (NWO), The Netherland.
"""
Name: diffusion class
Author: Nan Lin (sentient-codebot)
Date: Nov 2024

"""
from functools import partial
from typing import Callable, Iterator

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import reduce
from ema_pytorch import EMA
from torch import Tensor, nn
from tqdm import tqdm

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.energydiff._diffusion_base import (
    BetaScheduleType,
    DiffusionBase,
    LossType,
    ModelMeanType,
    ModelPrediction,
    ModelVarianceType,
    extract,
    get_beta_schedule,
    get_loss_weight,
)
from opensynth.models.energydiff.model import DenoisingTransformer
from opensynth.models.energydiff.sampler import DPMSolverSampler


class GaussianDiffusion1D(nn.Module, DiffusionBase):
    def __init__(
        self,
        base_model: DenoisingTransformer,  # denoise model
        num_timestep: int = 1000,
        model_mean_type: ModelMeanType = ModelMeanType.NOISE,
        model_variance_type: ModelVarianceType = ModelVarianceType.FIXED_SMALL,
        loss_type: LossType = LossType.MSE,
        beta_schedule_type: BetaScheduleType = BetaScheduleType.COSINE,
    ):
        super().__init__()
        self._model = base_model
        self.dim_in = base_model.dim_in
        self.num_timestep = num_timestep
        self._model_mean_type = model_mean_type
        self._model_variance_type = model_variance_type
        self.loss_type = loss_type
        self.beta_schedule_type = beta_schedule_type
        self.dpm_sampler = None

        # Check arguments
        if not isinstance(model_mean_type, ModelMeanType):
            raise ValueError(
                "objective must be ModelMeanType.X_START, \
                ModelMeanType.NOISE, ModelMeanType.V"
            )
        if not isinstance(model_variance_type, ModelVarianceType):
            raise ValueError("model variance type must be ModelVarianceType")
        if not isinstance(beta_schedule_type, BetaScheduleType):
            raise ValueError(
                "type_beta_schedule must be BetaScheduleType.\
                linear or BetaScheduleType.cosine"
            )
        if (
            not (self.model_variance_type == ModelVarianceType.LEARNED_RANGE)
            == self.model.learn_variance
        ):
            raise ValueError(
                "denoising_model.learn_variance must be \
                consistent with diffusion_model_variance_type"
            )

        if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
            raise NotImplementedError("Learned range is not implemented yet.")
        if self.loss_type.is_vb():
            raise NotImplementedError("KL loss is not implemented yet.")

        # calculate beta_schedule
        beta_schedule = get_beta_schedule(beta_schedule_type, num_timestep)

        # other diffusion q(x_t | x_{t-1}) needed coef
        alpha = 1.0 - beta_schedule
        alpha_cumprod = torch.cumprod(
            alpha, dim=0
        )  # shape: (num_timestep,) cumulative product
        alpha_cumprod_prev = F.pad(
            alpha_cumprod[:-1], (1, 0), value=1.0
        )  # replace beginning with 1.

        self.beta_schedule = beta_schedule
        self._alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev

        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.log_one_minus_alpha_cumprod = torch.log(1.0 - alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.rsqrt(alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod - 1.0)

        # coef for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            beta_schedule * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        )
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.cat(
            [posterior_variance[1:2], posterior_variance[1:]], dim=0
        ).log()
        # clipped because variance is 0 at the beginning
        self.posterior_mean_coef1 = (
            beta_schedule
            * torch.sqrt(alpha_cumprod_prev)
            / (1.0 - alpha_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alpha_cumprod_prev)
            * torch.sqrt(alpha)
            / (1.0 - alpha_cumprod)
        )

        # calculate loss weights
        loss_weight = get_loss_weight(alpha_cumprod, model_mean_type)

        # convert to float32 for gpu
        self.beta_schedule = self.beta_schedule.to(torch.float32)
        self._alpha_cumprod = self._alpha_cumprod.to(torch.float32)
        self.alpha_cumprod_prev = self.alpha_cumprod_prev.to(torch.float32)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(torch.float32)
        self.sqrt_one_minus_alpha_cumprod = (
            self.sqrt_one_minus_alpha_cumprod.to(torch.float32)
        )
        self.log_one_minus_alpha_cumprod = self.log_one_minus_alpha_cumprod.to(
            torch.float32
        )
        self.sqrt_recip_alpha_cumprod = self.sqrt_recip_alpha_cumprod.to(
            torch.float32
        )
        self.sqrt_recipm1_alpha_cumprod = self.sqrt_recipm1_alpha_cumprod.to(
            torch.float32
        )
        self.posterior_variance = self.posterior_variance.to(torch.float32)
        self.posterior_log_variance_clipped = (
            self.posterior_log_variance_clipped.to(torch.float32)
        )
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(torch.float32)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(torch.float32)
        self.loss_weight = loss_weight.to(torch.float32)

    @property
    def alpha_cumprod(self) -> torch.Tensor:
        return self._alpha_cumprod

    @property
    def model_mean_type(self) -> ModelMeanType:
        return self._model_mean_type

    @property
    def model_variance_type(self) -> ModelVarianceType:
        return self._model_variance_type

    @property
    def model(self) -> nn.Module:
        return self._model

    def predict_start_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ):
        return (
            extract(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            extract(self.sqrt_recip_alpha_cumprod, t, x_t.shape) * x_t
            - x_start
        ) / extract(self.sqrt_recipm1_alpha_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alpha_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
            * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alpha_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape) * v
        )

    def q_mean_variance(
        self,
        x_start: Tensor,
        t: Tensor,
    ):
        "forward x_0 -> x_t, q(x_t | x_0)"
        mean = extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
        variance = extract(
            self.sqrt_one_minus_alpha_cumprod**2, t, x_start.shape
        )
        log_variance = extract(
            self.log_one_minus_alpha_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_posterior_mean_variance(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
    ):
        "posterior q(x_{t-1} | x_t, x_0)"
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(
            self.posterior_variance, t, x_t.shape
        )  # shape: (batch, 1, 1)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: None | Tensor = None,
    ) -> Tensor:
        "sample x_t from q(x_t | x_0)"
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)
            * noise
        )

    def model_prediction(
        self,
        x_t: Tensor,
        t: Tensor,
        clip_x_start: bool = False,
    ) -> ModelPrediction:
        "get ModelPrediction(noise, x_start, var_factor)"
        model_output = self.model(x_t, t)
        _clip_fn: Callable = partial(torch.clamp, min=-1.0, max=1.0)
        _identity_fn: Callable = lambda x: x
        maybe_clip = _clip_fn if clip_x_start else _identity_fn

        # if learned variance range, split the mean and variance prediction
        if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
            if not model_output.shape == (
                x_t.shape[0],
                x_t.shape[1],
                2 * x_t.shape[2],
            ):
                raise ValueError(
                    "model output shape does not match expected shape."
                )

            model_output, model_var_factor = torch.split(
                model_output, x_t.shape[2], dim=-1
            )
        else:
            model_var_factor = None

        # calculate noise and x_start depending on model_mean_type
        if self.model_mean_type == ModelMeanType.NOISE:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise)
            x_start = maybe_clip(x_start)
            # optional, rederive noise from x_start after clipping
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        elif self.model_mean_type == ModelMeanType.X_START:
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        elif self.model_mean_type == ModelMeanType.V:
            v = model_output
            x_start = self.predict_start_from_v(x_t, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        else:
            raise NotImplementedError(
                f"model_mean_type: {self.model_mean_type} not implemented."
            )

        return ModelPrediction(pred_noise, x_start, model_var_factor)

    def p_mean_variance(
        self,
        x_t: Tensor,
        t: Tensor,
        clip_denoised: bool = False,
        model_kwargs=None,
    ) -> dict:
        """Get approximate mean and var of posterior p_model(x_{t-1} | x_t).
            Also returns x_start
        :param x_t: the input x_t, float, shape: (batch, sequence, dim)
        :param t: the time step, int, shape: (batch,)
        :param clip_denoised: whether to
            clip the denoised x_start to [-1, 1]
        :param model_kwargs: additional kwargs for \
            for model_prediction function

        :return dict: {
            'model_mean': model_mean
            'model_variance': model_variance
            'model_log_variance': model_log_variance
            'pred_x_start': pred_x_start
        }
        """
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        pred = self.model_prediction(x_t, t, **model_kwargs)
        pred_x_start = pred.pred_x_start
        var_factor = pred.pred_var_factor

        # get variance
        if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
            min_log = extract(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            max_log = extract(torch.log(self.beta_schedule), t, x_t.shape)
            frac = (var_factor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            model_variance = model_log_variance.exp()
        else:
            model_variance, model_log_variance = {
                ModelVarianceType.FIXED_LARGE: (
                    torch.cat(
                        [self.posterior_variance[1:2], self.beta_schedule[1:]],
                        dim=0,
                    ),
                    torch.log(
                        torch.cat(
                            [
                                self.posterior_variance[1:2],
                                self.beta_schedule[1:],
                            ],
                            dim=0,
                        )
                    ),
                ),
                ModelVarianceType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_variance_type]
            model_log_variance = extract(model_log_variance, t, x_t.shape)
            model_variance = extract(model_variance, t, x_t.shape)

        # get mean
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, min=-1.0, max=1.0)

        model_mean, *_ = self.q_posterior_mean_variance(pred_x_start, x_t, t)

        if not (
            model_mean.shape
            == model_log_variance.shape
            == model_variance.shape
            == x_t.shape
        ):
            raise ValueError(
                f"inconsistent shape: {model_mean.shape}, \
                {model_log_variance.shape}, {model_variance.shape}, \
                    {x_t.shape}"
            )

        return {
            "model_mean": model_mean,
            "model_variance": model_variance,
            "model_log_variance": model_log_variance,
            "pred_x_start": pred_x_start,
        }

    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        t: int,  # int, scalar, not a batched tensor.
        clip_denoised: bool = False,
        model_kwargs: None | dict = None,
        noise: None | Tensor = None,
    ) -> Tensor:
        """Apply p_model(x_{t-1} | x_t) to sample x_{t-1} from x_t.
        A inference-centric function, so `t` is an integer,
            same for every sample in the batch.
        :param x_t: x_t, shape (batch, channel, sequence)
        :param t: timestep, int
        :param clip_denoised: whether to clip the denoised x_{t-1} to [-1, 1]
        :param model_kwargs: kwargs for model_prediction
        :param noise: optional. noise to use
            for sampling, shape (batch, channel, sequence)

        :return: {
            'pred_x_prev': pred_x_prev,
            'pred_x_start': out['pred_x_start'],
        }
        """
        b, *_ = x_t.shape
        batched_time = torch.full((b,), t, device=x_t.device, dtype=torch.long)
        out_dict = self.p_mean_variance(
            x_t,
            t=batched_time,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        if noise is None or noise.shape != x_t.shape:
            noise = torch.randn_like(x_t) if t > 0 else 0.0
            # if t == 0, noise is not needed
        pred_x_prev = (
            out_dict["model_mean"]
            + (0.5 * out_dict["model_log_variance"]).exp() * noise
        )

        return {
            "pred_x_prev": pred_x_prev,
            "pred_x_start": out_dict["pred_x_start"],
        }

    @torch.no_grad()
    def p_sample_loop_progressive(
        self,
        shape: torch.Size,  # 'b l d'
        noise: None | Tensor = None,
        clip_denoised: bool = False,
        model_kwargs: None | dict = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Creates a generator that progressively samples from the diffusion \
            process from x_T to x_0.

        This method implements the progressive sampling loop that generates \
            samples by iteratively
        denoising from timestep T-1 to 0. At each step, it applies the \
            p_sample function to
        predict the previous timestep's sample.

        Args:
            shape (torch.Size): The shape of the tensor to sample, \
                expected in format 'b l d'
                (batch, length, dimension).
            noise (Optional[Tensor], optional): Initial noise tensor. \
                If None, random noise will be used.
                Defaults to None.
            clip_denoised (bool, optional): Whether to clip the denoised \
                values. Defaults to False.
            model_kwargs (Optional[dict], optional): Additional arguments \
                to pass to the model.
                Defaults to None.

        Yields:
            dict[str, Tensor]: Dictionary containing sampling information at \
                each timestep, including:
                - pred_x_prev: The predicted sample for the previous timestep
                - pred_noise: The predicted noise component
                - Other model-specific outputs

        Returns:
            Iterator[dict[str, Tensor]]: Generator yielding progressive \
                sampling results from T-1 to 0.
        """
        device = next(self.model.parameters()).device
        model_kwargs = model_kwargs

        x_t = noise if noise is not None else torch.randn(shape, device=device)
        # the initial x_t, t = T

        # yielding T-1 -> 0
        for t in tqdm(
            reversed(range(0, self.num_timestep)),
            desc="Sampling Loop Timestep",
            total=self.num_timestep,
        ):
            sample_out = self.p_sample(
                x_t,
                t=t,  # int
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )  # the noise parameter
            # IS NOT THE SAME AS the noise here.
            yield sample_out
            x_t = sample_out["pred_x_prev"]

    def p_sample_loop(
        self,
        shape: torch.Size,  # 'b l d'
        noise: None | Tensor = None,
        clip_denoised: bool = False,
        model_kwargs: None | dict = None,
    ) -> Tensor:
        "return the final x_0"
        for sample_out in self.p_sample_loop_progressive(
            shape, noise, clip_denoised, model_kwargs
        ):
            sample = sample_out["pred_x_prev"]

        return sample

    def _vb_terms_bpd(self, *args, **kwargs):
        raise NotImplementedError(
            "Not really useful. Do not use learned range."
        )

    def train_losses(
        self,
        x_start: Tensor,  # true x_0
        t: Tensor,  # timestep
        noise: None | Tensor = None,  # noise
        model_kwargs: None | dict = None,  # kwargs for model_prediction
    ) -> dict[str, Tensor]:
        """return batched loss. not yet averaged over batch.

        given
        - x_start: true data samples, shape (batch, channel, sequence)
        - t: randomly sampled (diffusion) timestep, shape (batch,)
        - noise: optional, noise to
            impose on x_start, shape (batch, channel, sequence)

        apply forward diffusion to get x_t;
        apply model to get approximate posterior parameters;
        calculate loss on the posterior parameters.

        :return dict: loss_terms: {
            'loss': shape (batch,)
            'mse': shape (batch,), if applicable
            'vb': vb, shape (batch,), if applicable
        }
        """
        # forward, get x_t
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        loss_terms = {}
        # get model prediction
        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            raise NotImplementedError("KL loss is not implemented yet.")
        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            out = self.p_mean_variance(
                x_t, t, clip_denoised=False, model_kwargs=model_kwargs
            )

            if self.model_variance_type == ModelVarianceType.LEARNED_RANGE:
                raise NotImplementedError(
                    "Learned range is not implemented yet."
                )

            # mse terms
            target = {
                ModelMeanType.NOISE: noise,
                ModelMeanType.X_START: x_start,
                ModelMeanType.V: self.predict_v(x_start, t, noise),
            }[self.model_mean_type]
            pred_noise = self.predict_noise_from_start(
                x_t, t, out["pred_x_start"]
            )
            source = {
                ModelMeanType.NOISE: pred_noise,
                ModelMeanType.X_START: out["pred_x_start"],
                ModelMeanType.V: self.predict_v(
                    out["pred_x_start"], t, pred_noise
                ),
            }[self.model_mean_type]
            if not target.shape == source.shape:
                raise ValueError("target and source shape mismatch.")

            mse_loss = reduce(
                F.mse_loss(source, target, reduction="none"),
                "b ... -> b",
                "mean",
            )  # shape (batch, )
            loss_terms["mse"] = mse_loss
            loss_terms["loss"] = mse_loss
        else:
            raise ValueError(
                "loss_type must be one of MSE, RESCALED_MSE, KL, RESCALED_KL"
            )

        return loss_terms

    def forward(
        self,
        x_start: Tensor,  # true x_0
        noise: None | Tensor = None,  # noise
    ) -> dict[str, Tensor]:
        "return scalar loss. averaged over batch already."
        B, L, D = x_start.shape
        device = x_start.device
        if not D == self.dim_in:
            raise ValueError(f"Dimension mismatch: {D} != {self.dim_in}")

        t = torch.randint(0, self.num_timestep, (B,), device=device)
        loss_terms = self.train_losses(x_start, t, noise)

        for k, v in loss_terms.items():
            loss_terms[k] = v.mean()

        return loss_terms

    # two functions we use to sample.
    def ancestral_sample(
        self,
        num_sample: int,
        batch_size: int,
        data_sequence_length: int,
        data_feature_dim: int,  # 1 for univariate, 2 for bivariate
        # cond: torch.Tensor|None=None,
        # cfg_scale: float = 1.,
    ) -> torch.Tensor:
        """sample from either given diffusion model.

        Arguments:
            - num_sample: int, number of samples to generate
            - batch_size: int, batch size for sampling
            - cond: shape [batch_size, channel, 1], condition for sampling
        """
        # process cond
        # if cond is not None:
        #     if model is not None:
        #         _device = next(model.parameters()).device
        #         cond = cond.to(_device)
        #     else:
        #         cond = cond.to(trainer.device)

        # process batch size split
        if num_sample < batch_size:
            list_batch_size = [num_sample]
        else:
            list_batch_size = [batch_size] * (num_sample // batch_size)
            if num_sample % batch_size != 0:
                list_batch_size.append(num_sample % batch_size)

        # sample
        list_sample = []
        for idx, batch_size in enumerate(list_batch_size):
            print(
                f"sampling batch {idx + 1}/{len(list_batch_size)}, \
                    batch size {batch_size}. "
            )
            sample_batch = self.p_sample_loop(
                shape=(batch_size, data_sequence_length, data_feature_dim),
                noise=None,
                clip_denoised=False,  # True requires data scaling to (-1,1)
            )

            list_sample.append(sample_batch)
        all_sample = torch.cat(list_sample, dim=0)

        return all_sample

    def _init_dpm_sampler(self):
        self.dpm_sampler = DPMSolverSampler(self)

    def dpm_solver_sample(
        self,
        total_num_sample: int,
        batch_size: int,
        step: int,
        shape: tuple[int, int],
        # conditioning: torch.Tensor|None,
        # cfg_scale: float,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        if step < 15:
            raise ValueError(
                "sampling step must be at least 15 due to\
                constraint of DPM-Solver."
            )
        if self.dpm_sampler is None:
            self._init_dpm_sampler()
        assert self.dpm_sampler is not None

        num_sample = total_num_sample
        if num_sample < batch_size:
            list_batch_size = [num_sample]
        else:
            list_batch_size = [batch_size] * (num_sample // batch_size)
            if num_sample % batch_size != 0:
                list_batch_size.append(num_sample % batch_size)

        list_sample = []
        for idx, batch_size in enumerate(list_batch_size):
            print(
                f"sampling batch {idx + 1}/{len(list_batch_size)}, \
                    batch size {batch_size}. "
            )
            sample_batch, _ = self.dpm_sampler.sample(
                S=step,
                batch_size=batch_size,
                shape=shape,
                # conditioning = conditioning,
                # cfg_scale = cfg_scale,
            )
            list_sample.append(sample_batch)

        all_sample = torch.cat(list_sample, dim=0)

        if clip_denoised:
            all_sample = torch.clamp(all_sample, -1, 1)
        return all_sample


class PLDiffusion1D(pl.LightningModule):
    "Pytorch Lightning wrapper for GaussianDiffusion1D"

    def __init__(
        self,
        # transformer model arguments
        dim_base: int = 256,
        dim_in: int = 1,
        num_attn_head: int = 4,
        num_decoder_layer: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        learn_variance: bool = False,
        # diffusion model arguments
        num_timestep: int = 1000,
        model_mean_type: ModelMeanType = ModelMeanType.NOISE,
        model_variance_type: ModelVarianceType = ModelVarianceType.FIXED_SMALL,
        loss_type: LossType = LossType.MSE,
        beta_schedule_type: BetaScheduleType = BetaScheduleType.COSINE,
        # PL optimizer arguments
        lr: float = 1e-4,
        ema_update_every: int = 5,
        ema_decay: float = 0.9999,
        disable_init_proj: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        base_model = DenoisingTransformer(
            dim_base=dim_base,
            dim_in=dim_in,
            num_attn_head=num_attn_head,
            num_decoder_layer=num_decoder_layer,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learn_variance=learn_variance,
            disable_init_proj=disable_init_proj,
        )
        self.diffusion_model = GaussianDiffusion1D(
            base_model=base_model,
            num_timestep=num_timestep,
            model_mean_type=model_mean_type,
            model_variance_type=model_variance_type,
            loss_type=loss_type,
            beta_schedule_type=beta_schedule_type,
        )
        self.ema = EMA(
            self.diffusion_model,
            beta=ema_decay,
            update_every=ema_update_every,
            include_online_model=True,
        )
        self.lr = lr
        self.ema_update_every = ema_update_every
        self.ema_decay = ema_decay

    # setup function
    def setup(self, stage: str) -> None:
        pass  # nothing to setup

    def on_before_batch_transfer(
        self,
        batch: TrainingData,
        dataloader_idx: int,
    ):
        kwh_data = batch["kwh"]  # shape: (batch, sequence)
        kwh_data = kwh_data.unsqueeze(-1)  # shape: (batch, sequence, 1)
        return kwh_data

    # training step
    def training_step(
        self,
        batch: Tensor,
        batch_idx: int,  # -> step counter
    ) -> Tensor:
        loss_terms = self.diffusion_model(x_start=batch)
        loss = loss_terms["loss"]
        mse = loss_terms["mse"].item() if "mse" in loss_terms else 0.0
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_mse_loss",
            mse,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()

    # validation step
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss_terms = self.diffusion_model(x_start=batch)
        loss = loss_terms["loss"]
        mse = loss_terms["mse"].item() if "mse" in loss_terms else 0.0
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_mse_loss",
            mse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    # test step
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step is not implemented yet.")

    # configure optimizer
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.diffusion_model.parameters(), lr=self.lr
        )  # just online model
        return optimizer
