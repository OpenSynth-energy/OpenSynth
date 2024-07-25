# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import torch.nn.functional as F


def MMDLoss(y: torch.tensor, x: torch.tensor) -> torch.tensor:
    """
    Calculate MMD Loss

    Args:
        y (torch.tensor): Tensor Y
        x (torch.tensor): Tensor X

    Returns:
        torch.tensor: MMD Distances between Tensor Y and X
    """

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
    y_pred: torch.tensor, y_real: torch.tensor, quantile: float
) -> torch.tensor:
    """
    Calculate quantile loss
    #TODO: Test this function
    Args:
        y_pred (torch.tensor): Predicted quantile
        y_real (torch.tensor): Actual quantile
        quantile (float): Quantile value

    Returns:
        torch.tensor: Quantile loss
    """
    return torch.mean(
        torch.max(
            quantile * (y_real - y_pred), (1 - quantile) * (y_pred - y_real)
        )
    )


def calculate_training_loss(
    x_hat: torch.tensor,
    x: torch.tensor,
    mse_weight: float,
    quantile_upper_weight: float,
    quantile_lower_weight: float,
    quantile_median_weight: float,
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    """
    Calculate training losses for Faraday.
    Losses are a combined total of:
    1) MMD loss
    2) MSE loss * mse_weight
    3) upper quantile loss * upper quantile weight
    4) lower quantile loss * lower quantile weight
    5) median quantile loss * median quantile weight
    Args:
        x_hat (torch.tensor): Generated tensor
        x (torch.tensor): Real tensor
        mse_weight (float): MSE Weight
        quantile_upper_weight (float): Upper quantile weight
        quantile_lower_weight (float): Lower quantile weight
        quantile_median_weight (float): Median quantile weight
        lower_quantile (float): Lower quantile value
        upper_quantile (float): Upper quantile value

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        Total loss, MMD loss, MSE loss, Quantile loss
    """
    mmd_loss = MMDLoss(x_hat, x)
    mse_loss = F.mse_loss(x_hat, x) * mse_weight
    quantile_upper_loss = (
        quantile_loss(x_hat, x, upper_quantile) * quantile_upper_weight
    )
    quantile_lower_loss = (
        quantile_loss(x_hat, x, lower_quantile) * quantile_lower_weight
    )
    quantile_median_loss = (
        quantile_loss(x_hat, x, 0.5) * quantile_median_weight
    )
    quantile_losses = (
        quantile_upper_loss + quantile_lower_loss + quantile_median_loss
    )

    total_loss = mmd_loss + mse_loss + quantile_losses

    return total_loss, mmd_loss, mse_loss, quantile_losses
