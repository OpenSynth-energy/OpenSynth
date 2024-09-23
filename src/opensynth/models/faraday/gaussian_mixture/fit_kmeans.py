# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from opensynth.models.faraday.gaussian_mixture.kmeans_lightning import (
    KMeansLightningModule,
    KmeansRandomInitLightningModule,
)
from opensynth.models.faraday.gaussian_mixture.model import KMeansModel

logger = CSVLogger("lightning_logs", name="kmeans_logs")


def fit_kmeans(
    data: DataLoader,
    num_components: int,
    vae_module: pl.LightningModule,
    input_dim: int,
    max_epochs: int = 500,
    convergence_tolerance: float = 1e-4,
    accelerator: str = "mps",
    devices: int = 1,
) -> torch.Tensor:
    """Fit K-means model to data using PyTorch Lightning

    Function relies on KMeansModel, KMeansLightningModule, and
        KmeansRandomInitLightningModule from
        opensynth.models.faraday.gaussian_mixture.model,
        opensynth.models.faraday.gaussian_mixture.kmeans_lightning.

    Args:
        data (DataLoader): training data
        num_components (int): number of components or clusters in the data
        vae_module (pl.LightningModule): A trained VAE model
        input_dim (int): input dimensions of data
        max_epochs(int): maximum epochs for K-means fitting
        convergence_tolerance(float): convergence tolerance for early stopping
            of K-means training
        accelerator (str): accelerator for Pytorch Lightning
        devices (int): number of devices (GPUs) to use.

    Returns:
        torch.Tensor: k-means centroids
    """
    # Initiate K-means model
    kmeans_model_ = KMeansModel(
        num_clusters=num_components, num_features=input_dim
    )

    # Use uniform distribution to get initial centroids
    init_module = KmeansRandomInitLightningModule(
        kmeans_model_, vae_module, num_components, input_dim
    )
    trainer = pl.Trainer(
        max_epochs=1, accelerator=accelerator, devices=devices, logger=logger
    )  # setting initial values - run for 1 epoch
    trainer.fit(init_module, data)

    # Fit K-means
    kmeans_module = KMeansLightningModule(
        kmeans_model_,
        vae_module,
        num_components,
        input_dim,
        convergence_tolerance,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
    )
    trainer.fit(kmeans_module, data)

    return kmeans_model_.centroids
