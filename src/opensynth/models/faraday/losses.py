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
    return torch.cat([x[i].repeat(weights[i], 1) for i in range(len(weights))])


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
    if sample_weights:
        x = _expand_samples(x, sample_weights)
        y = _expand_samples(y, sample_weights)

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

    return torch.mean(XX + YY - 2.0 * XY)


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

    quantile_loss = torch.max(
        quantile * (y_real - y_pred), (1 - quantile) * (y_pred - y_real)
    )

    if sample_weights is not None:
        quantile_loss = quantile_loss * sample_weights
        return torch.sum(quantile_loss) / torch.sum(sample_weights)

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
    squared_loss = F.mse_loss(y_pred, y_real, reduction="none")

    if sample_weights is not None:
        squared_loss = squared_loss * sample_weights
        return torch.sum(squared_loss) / torch.sum(sample_weights)

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
    mse_loss_ = mse_loss(x_hat, x, sample_weights=sample_weights)
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
