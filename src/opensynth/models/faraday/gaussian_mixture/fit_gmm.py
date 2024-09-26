# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import time
from typing import Tuple

import numpy
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from opensynth.models.faraday.gaussian_mixture.fit_kmeans import fit_kmeans
from opensynth.models.faraday.gaussian_mixture.gmm_lightning import (
    GaussianMixtureInitLightningModule,
    GaussianMixtureLightningModule,
)
from opensynth.models.faraday.gaussian_mixture.model import (
    GaussianMixtureModel,
)
from opensynth.models.faraday.vae_model import FaradayVAE

logger = CSVLogger("lightning_logs", name="gmm_logs")


def fit_gmm(
    data: DataLoader,
    num_components: int,
    vae_module: FaradayVAE,
    num_features: int,
    covariance_type: str = "full",
    gmm_max_epochs: int = 10000,
    gmm_convergence_tolerance: float = 1e-6,
    covariance_regularization: float = 1e-6,
    initial_regularization: float = 1e-6,
    init_method: str = "kmeans",
    kmeans_max_epochs: int = 500,
    kmeans_convergence_tolerance: float = 1e-4,
    is_batch_training: bool = True,
    accelerator: str = "cpu",
    devices: int = 1,
) -> Tuple[GaussianMixtureLightningModule, pl.Trainer, GaussianMixtureModel]:
    """Fit Gaussian Mixture Model to data using PyTorch Lightning

    Args:
        data (DataLoader): training dataset
        num_components (int): number of Gaussian components in the
            mixture model.
        vae_module (FaradayVAE): trained VAE model.
        num_features (int): number of features in latent space
            (size of latent space + number of non encoded features)
        covariance_type (str, optional): GMM covariance type.
            Defaults to "full".
        gmm_max_epochs (int, optional): maximum epochs to run GMM fitting.
            Defaults to 10000.
        gmm_convergence_tolerance (float, optional): convergence tolerance for
            early stopping of GMM training. Early stopping happens when the
            negative log probability doesn't change more than this value.
            Defaults to 1e-6.
        covariance_regularization (float, optional): a small value which is
            added to the diagonal of the covariance matrix to ensure that it is
            positive semi-definite. Defaults to 1e-6.
        initial_regularization (float, optional): Regularization factor applied
            during the initialization of the GMM. Defaults to 1e-6. Increasing
            this value can help prevent singular covariance matrices during
            initialization but may lead to a poorer initial solution.
        init_method (str, optional): initialisation method for GMM. Allowed
            "rand" or "kmeans". Defaults to "kmeans".
        kmeans_max_epochs (int, optional): maximum epochs to run k-means
            fitting if init_method = "kmeans". Defaults to 500.
        kmeans_convergence_tolerance (float, optional): convergence tolerance
            for early stopping of k-means training. Defaults to 1e-4.
        is_batch_training (bool, optional): flag whether batch training.
            Defaults to True.
        accelerator (str, optional): accelerator for training.
            Defaults to "cpu".
        devices (int, optional): number of devices (or GPUs) to run training.
            Defaults to 1.

    Returns:
        GaussianMixtureModel: GMM model
        GaussianMixtureLightningModule: GMM lightning module
        pl.Trainer: Pytorch Lightning Trainer for GMM
    """

    start_time = time.time()

    # Initialize the GMM model
    model_ = GaussianMixtureModel(
        covariance_type, num_components, num_features
    )

    if init_method == "kmeans":
        print("Running K-Means initialisation")
        print("--------------------------------------------------------------")
        # Set initial means for GMM using result from K-means fitting
        centroids = fit_kmeans(
            data,
            num_components,
            vae_module,
            num_features,
            kmeans_max_epochs,
            kmeans_convergence_tolerance,
            accelerator,
            devices,
        )
        # Use k-means centroids as initial means for GMM
        model_.means.copy_(centroids)
        print("Done K-Means initialisation")
        max_epochs_init = 1
    elif init_method == "rand":
        # Use random initialization for GMM
        max_epochs_init = 1 + int(is_batch_training)

    print("Beginning GMM Training")
    # Initialise GMM model using InitLightningModule
    init_module = GaussianMixtureInitLightningModule(
        model_,
        vae_module,
        num_components=num_components,
        num_features=num_features,
        init_method=init_method,
        covariance_type=covariance_type,
        covariance_regularization=initial_regularization,
        is_batch_training=is_batch_training,
    )

    pl.Trainer(
        max_epochs=max_epochs_init,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    ).fit(init_module, data)

    # Run GMM fitting
    gmm_module = GaussianMixtureLightningModule(
        model_,  # init_module.model,
        vae_module,
        num_components,
        num_features,
        is_batch_training=is_batch_training,
        covariance_type=covariance_type,
        covariance_regularization=covariance_regularization,
        convergence_tolerance=gmm_convergence_tolerance,
    )
    trainer = pl.Trainer(
        max_epochs=gmm_max_epochs,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(gmm_module, data)

    delta_time = time.time() - start_time
    print(f"Total training time: {delta_time}")

    if not (numpy.isclose(model_.component_probs.sum(), 1.0)):
        raise (
            ValueError(
                "Gaussian mixture component probabilities do not sum to 1.0"
            )
        )

    return gmm_module, trainer, model_
