# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
#
# The EnergyDiff model is made available via Nan Lin and Pedro P. Vergara
# from the Delft University of Technology. Nan Lin and Pedro P. Vergara are
# funded via the ALIGN4Energy Project (with project number NWA.1389.20.251) of
# the research programme NWA ORC 2020 which is (partly) financed by the Dutch
# Research Council (NWO), The Netherland.
"""
Name: diffusion base class and utility functions
Author: Nan Lin (sentient-codebot)
Date: Nov 2024

"""
import enum
import math
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Iterator

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "pred_var_factor"]
)


class ModelMeanType(enum.Enum):
    X_START = "x_start"
    NOISE = "noise"
    V = "v"


class ModelVarianceType(enum.Enum):
    FIXED_SMALL = "fixed_small"  # default, necessary for dpm-solver
    FIXED_LARGE = "fixed_large"
    LEARNED_RANGE = "learned_range"


class LossType(enum.Enum):
    MSE = "mse"
    RESCALED_MSE = "rescaled_mse"
    KL = "kl"
    RESCALED_KL = "rescaled_kl"

    def is_vb(self):
        return self in (LossType.KL, LossType.RESCALED_KL)


class BetaScheduleType(enum.Enum):
    DISCRETE = "discrete"
    LINEAR = "linear"
    COSINE = "cosine"  # default


def extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    """Extract values from tensor 'a' at positions specified by tensor 't' \
        and reshape to 'x_shape'.

    This function extracts values from tensor 'a' at positions specified \
        by tensor 't',
    and reshapes the result to match the shape specified by 'x_shape'.

    Args:
        a (Tensor): Source tensor to extract values from. 1D tensor.
        t (Tensor): Tensor containing positions where values should be \
            extracted. (batch,) or (batch, 1) tensor.
        x_shape (torch.Size): Target shape for the output tensor.

    Returns:
        Tensor: Extracted values reshaped to match x_shape.

    Example:
        >>> a = torch.tensor([0.1, 0.2, 0.3, 0.4])
        >>> t = torch.tensor([1, 3])  # extract values at indices 1 and 3
        >>> x_shape = (2, 3, 2)  # target shape
        >>> result = extract(a, t, x_shape)
        >>> result.shape
        torch.Size([2, 3, 2])
        >>> result[0, :, :]  # filled with 0.2 (a[1])
        tensor([[0.2, 0.2],
                [0.2, 0.2],
                [0.2, 0.2]])
        >>> result[1, :, :]  # filled with 0.4 (a[3])
        tensor([[0.4, 0.4],
                [0.4, 0.4],
                [0.4, 0.4]])

    Note:
        The function automatically converts tensor 'a' to float32 dtype and
        matches the device of tensor 't'.
    """
    # shape chec
    if t.dim() > 2 or (t.dim() == 2 and t.shape[1] != 1):
        raise ValueError("t must be a 1D tensor or 2D tensor with shape (B,1)")

    if t.dim() == 2:
        t = t.squeeze(-1)  # Convert 2D (B,1) to 1D (B,)

    if t.shape[0] != x_shape[0]:
        raise ValueError("t and x_shape must have the same batch size")

    # convert a to float32 and match device
    a = a.to(device=t.device, dtype=torch.float32)  # !dtype is fixed here
    b, *_ = t.shape
    dim_x = len(x_shape)
    # extract values from a at positions specified by t
    out = a.gather(-1, t)
    # fill missing dimensions with singleton dimensions (braodcast)
    target_shape = (b, *(1 for _ in range(dim_x - 1)))
    # using reshape to covert to (b, 1, ..., 1)
    # and use "+" to broadcast to target shape
    out = out.reshape(*target_shape) + torch.zeros(x_shape, device=t.device)

    return out


def linear_beta_schedule(num_timestep: int) -> Tensor:
    """get the linear beta schedule.

    Arguments:
        num_timestep -- int

    Returns:
        Tensor -- shape: (num_timestep,)
    """
    scale = 1000.0 / num_timestep
    beta_start = scale * 1e-4
    beta_end = scale * 2e-2
    return torch.linspace(
        beta_start, beta_end, num_timestep, dtype=torch.float64
    )  # shape: (num_timestep,)


def cosine_beta_schedule(num_timestep: int, s: float = 0.008) -> Tensor:
    "as proposed in https://openreview.net/forum?id=-NEXDKk8gZ"
    num_step = num_timestep + 1
    x = torch.linspace(
        0, num_timestep, num_step, dtype=torch.float64
    )  # shape: (num_step,)
    alpha_cumprod = (
        torch.cos((x / num_timestep + s) / (1 + s) * math.pi / 2) ** 2
    )  # shape: (num_step,)
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]  # shouldn't [0] be 1.?
    beta_schedule = 1 - (
        alpha_cumprod[1:] / alpha_cumprod[:-1]
    )  # shape: (num_timestep,)
    return torch.clip(beta_schedule, 0, 0.999)  # shape: (num_timestep,)


def get_beta_schedule(
    beta_schedule_type: BetaScheduleType, num_timestep: int
) -> Tensor:
    if beta_schedule_type == BetaScheduleType.LINEAR:
        return linear_beta_schedule(num_timestep)
    elif beta_schedule_type == BetaScheduleType.COSINE:
        return cosine_beta_schedule(num_timestep)
    else:
        raise ValueError("type_beta_schedule must be one of linear, cosine")


def get_loss_weight(
    alpha_cumprod: Tensor, model_mean_type: ModelMeanType
) -> Tensor:
    # TODO unit test
    # calculate loss weights
    snr = alpha_cumprod / (1.0 - alpha_cumprod)
    if model_mean_type == ModelMeanType.NOISE:
        loss_weight = torch.ones_like(snr)
    elif model_mean_type == ModelMeanType.X_START:
        loss_weight = snr
    elif model_mean_type == ModelMeanType.V:
        loss_weight = snr / (snr + 1.0)

    return loss_weight


class DiffusionBase(ABC):
    @property
    @abstractmethod
    def alpha_cumprod(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def model_mean_type(self) -> ModelMeanType:
        pass

    @property
    @abstractmethod
    def model_variance_type(self) -> ModelVarianceType:
        pass

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        pass

    @abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        pass
