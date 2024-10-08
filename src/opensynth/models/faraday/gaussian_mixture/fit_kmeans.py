# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import torch
from pytorch_lightning.loggers import CSVLogger
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from opensynth.models.faraday.vae_model import FaradayVAE

logger = CSVLogger("lightning_logs", name="kmeans_logs")


def fit_kmeans(
    data: DataLoader,
    num_components: int,
    vae_module: FaradayVAE,
) -> torch.Tensor:
    """Fit K-means model to data using Sklearn

    # Using sklearn implementation of K-means as opposed to PyTorch Lightning
    # future versions may use PyTorch Lightning, if GPU acceleration is needed.

    Args:
        data (DataLoader): training data
        num_components (int): number of components or clusters in the data
        vae_module (FaradayVAE): trained VAE model

    Returns:
        torch.Tensor: k-means centroids
    """
    # TODO : perform k-means on a random subsample of the training dataset to
    # speed up convergence. This is necessary when working with large datasets.
    kmeans_model_ = KMeans(n_clusters=num_components)
    next_batch = next(iter(data))
    kwh = next_batch["kwh"]
    features = next_batch["features"]
    vae_input = vae_module.reshape_data(kwh, features)
    vae_output = vae_module.encode(vae_input)
    model_input = (
        vae_module.reshape_data(vae_output, features).detach().numpy()
    )
    kmeans_fit = kmeans_model_.fit(model_input)

    return torch.tensor(kmeans_fit.cluster_centers_)
