# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import torch.nn.functional as F


def _expand_samples(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Expand the repeat in the tensor based on the sample weights
    e.g. if x: [1,2,3] and weights: [1,2,3], the output will be [1,2,2,3,3,3]

    Args:
        x (torch.Tensor): Input tensor
        weights (torch.Tensor): Sample weight

    Returns:
        torch.Tensor: Repeated tensor by sample weight
    """
    weights = weights.squeeze().long()
    return torch.repeat_interleave(x, weights, dim=0)


def _check_shape(x: torch.Tensor) -> torch.Tensor:
    """
    Check that tensor is either a 1-D or 2_D tensor.
    If tensor is 1-D, reshape to [N,1] tensor.

    Args:
        x (torch.Tensor): Input tensor

    Raises:
        ValueError: Raises error if tensor is not 1-D or 2-D

    Returns:
        torch.Tensor: Reshaped tensor
    """
    if len(x.shape) == 1:
        return x.reshape(-1, 1)
    elif len(x.shape) == 2:
        return x
    else:
        raise ValueError(f"Input tensor {x} must be 1D or 2D")


def mmd_loss(
    y: torch.Tensor, x: torch.Tensor, sample_weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate MMD Loss

    Args:
        y (torch.Tensor): Tensor Y
        x (torch.Tensor): Tensor X

    Returns:
        torch.Tensor: MMD Distances between Tensor Y and X
    """
    # Expand the weights array based on number of samples
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros(xx.shape),
        torch.zeros(xx.shape),
        torch.zeros(xx.shape),
    )
    XX = XX.to(x)
    YY = YY.to(x)
    XY = XY.to(x)

    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    mmd_loss = XX + YY - 2.0 * XY

    if sample_weights is not None:
        weights = sample_weights.squeeze().long()
        mmd_loss = torch.repeat_interleave(mmd_loss, weights, dim=0)
        mmd_loss = mmd_loss * weights
        return torch.sum(mmd_loss) / torch.sum(sample_weights) ** 2

    return torch.mean(mmd_loss)


def quantile_loss(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    quantile: float,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Calculate quantile loss
    Args:
        y_pred (torch.Tensor): Predicted quantile
        y_real (torch.Tensor): Actual quantile
        quantile (float): Quantile value

    Returns:
        torch.Tensor: Quantile loss
    """
    y_pred = _check_shape(y_pred)
    y_real = _check_shape(y_real)
    quantile_loss = torch.max(
        quantile * (y_real - y_pred), (1 - quantile) * (y_pred - y_real)
    )

    if sample_weights is not None:
        quantile_loss = quantile_loss * sample_weights.reshape(
            len(sample_weights), 1
        )

        return torch.sum(quantile_loss) / (
            torch.sum(sample_weights) * y_pred.shape[1]
        )

    return torch.mean(quantile_loss)


def mse_loss(
    y_pred: torch.Tensor,
    y_real: torch.Tensor,
    sample_weights: torch.Tensor = None,
):
    """
    Calculate MSE loss. If sample weights are provided, sample losses are
    multiplied by the sample weight before averaging.

    Args:
        y_pred (torch.Tensor): Prediction
        y_real (torch.Tensor): Actual
        sample_weights (torch.Tensor, optional): Sample Weights.
        Defaults to None.

    Returns:
        _type_: _description_
    """
    y_pred = _check_shape(y_pred)
    y_real = _check_shape(y_real)
    squared_loss = F.mse_loss(y_pred, y_real, reduction="none")

    if sample_weights is not None:
        squared_loss = squared_loss * sample_weights.reshape(
            len(sample_weights), 1
        )
        return torch.sum(squared_loss) / (
            torch.sum(sample_weights) * y_pred.shape[1]
        )

    return torch.mean(squared_loss)


def calculate_training_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mse_weight: float,
    quantile_upper_weight: float,
    quantile_lower_weight: float,
    quantile_median_weight: float,
    lower_quantile: float,
    upper_quantile: float,
    sample_weights: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate training losses for Faraday.
    Losses are a combined total of:
    1) MMD loss
    2) MSE loss * mse_weight
    3) upper quantile loss * upper quantile weight
    4) lower quantile loss * lower quantile weight
    5) median quantile loss * median quantile weight
    Args:
        x_hat (torch.Tensor): Generated tensor
        x (torch.Tensor): Real tensor
        mse_weight (float): MSE Weight
        quantile_upper_weight (float): Upper quantile weight
        quantile_lower_weight (float): Lower quantile weight
        quantile_median_weight (float): Median quantile weight
        lower_quantile (float): Lower quantile value
        upper_quantile (float): Upper quantile value

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Total loss, MMD loss, MSE loss, Quantile loss
    """
    mmd_loss_ = mmd_loss(x_hat, x, sample_weights=sample_weights)
    mse_loss_ = mse_loss(x_hat, x, sample_weights=sample_weights) * mse_weight
    quantile_upper_loss = (
        quantile_loss(x_hat, x, upper_quantile, sample_weights)
        * quantile_upper_weight
    )
    quantile_lower_loss = (
        quantile_loss(x_hat, x, lower_quantile, sample_weights)
        * quantile_lower_weight
    )
    quantile_median_loss = (
        quantile_loss(x_hat, x, 0.5, sample_weights) * quantile_median_weight
    )
    quantile_losses = (
        quantile_upper_loss + quantile_lower_loss + quantile_median_loss
    )

    total_loss = mmd_loss_ + mse_loss_ + quantile_losses

    return total_loss, mmd_loss_, mse_loss_, quantile_losses
