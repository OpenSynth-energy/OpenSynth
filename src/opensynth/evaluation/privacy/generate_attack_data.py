# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import torch

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday import FaradayModel


def generate_synthetic_samples(
    model: FaradayModel, dm: LCLDataModule, n_samples: int
) -> torch.tensor:
    """
    Generate synthetic samples from model.
    TODO: Add support for non-Faraday models in the future by
    creating a parent abstract class to inherit from.

    Args:
        model (FaradayModel): Model
        dm (LCLDataModule): Data module.
        n_samples (int): Number of synhetic samples to generate.

    Returns:
        torch.tensor: Synthetic samples.
    """
    synthetic_samples = model.sample_gmm(n_samples)
    synthetic_kwh = synthetic_samples[0]
    synthetic_kwh = dm.reconstruct_kwh(synthetic_kwh)
    synthetic_kwh = torch.clip(synthetic_kwh, min=0)
    return synthetic_kwh


def draw_real_data(
    dm: LCLDataModule, n_samples: int, outliers: bool = False
) -> torch.tensor:
    """
    Draw real samples from data module

    Args:
        dm (LCLDataModule): Data module
        n_samples (int): Number of samples to draw.
        outliers (bool, optional): Return outliers instead of normal
        training data. Defaults to False.

    Returns:
        torch.tensor: Real training samples.
    """
    if outliers:
        samples = next(iter(dm.outlier_dataloader()))
    else:
        dm.batch_size = n_samples
        samples = next(iter(dm.train_dataloader()))

    real_kwh = samples[0]
    real_kwh = dm.reconstruct_kwh(real_kwh)
    real_kwh = torch.clip(real_kwh, min=0)
    return real_kwh
