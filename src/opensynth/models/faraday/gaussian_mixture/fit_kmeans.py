"""Script containing function to fit K-means model to data using
PyTorch Lightning. Code is based on the PyCave framework.
"""

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday.gaussian_mixture.kmeans_lightning import (
    KMeansLightningModule,
    KmeansRandomInitLightningModule,
)
from opensynth.models.faraday.gaussian_mixture.model import KMeansModel

logger = CSVLogger("logs", name="kmeans_logs")


def fit_kmeans(
    data: LCLDataModule,
    num_components: int,
    vae_module: pl.LightningModule,
    input_dim: int,
    max_epochs: int = 500,
    convergence_tolerance: float = 1e-4,
    accelerator: str = "mps",
    devices: int = 1,
    plot: bool = False,
) -> torch.Tensor:
    """Fit K-means model to data using PyTorch Lightning

    Function relies on KMeansModel, KMeansLightningModule, and
        KmeansRandomInitLightningModule from
        opensynth.models.faraday.gaussian_mixture.model,
        opensynth.models.faraday.gaussian_mixture.kmeans_lightning.

    Plot will only show if input_dim == 2.

    Args:
        data (LCLDataModule): training data
        num_components (int): number of components or clusters in the data
        vae_module (pl.LightningModule): A trained VAE model
        input_dim (int): input dimensions of data
        max_epochs(int): maximum epochs for K-means fitting
        convergence_tolerance(float): convergence tolerance for early stopping
            of K-means training
        accelerator (str): accelerator for Pytorch Lightning
        devices (int): number of devices (GPUs)
        plot(bool): display plot of centroid positions on data

    Raises:
        Warning: if plot == True and input_dim != 2

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


def plot_centroids(
    data: torch.Tensor, means: torch.Tensor, n_samples: int = 500
) -> None:
    """Plot centroid positions. Use to visualise results if input_dim == 2.

    Args:
        data (torch.Tensor): data across 2 dimensions
        means (torch.Tensor): centroid positions of shape [num_components, 2]
    """
    fig, ax = plt.subplots(1, 1)

    # Extract data from the synthetic Gaussian dataset
    x = data.gauss_dataset()
    if x.size(0) > n_samples:
        # Plot a sample of the original dataset
        x = x[torch.randperm(x.size(0))[:n_samples]]

    # Data points
    for i, point in enumerate(x):
        ax.scatter(*point, color="grey", alpha=0.6)

    # Plot centroids
    for point in means:
        ax.scatter(*point, color="red", marker="x")

    plt.tight_layout()
    plt.show()
    plt.close()
