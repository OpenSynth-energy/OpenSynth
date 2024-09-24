# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""
from typing import List, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torchmetrics import MeanMetric

from opensynth.models.faraday.gaussian_mixture.metrics import (
    CentroidAggregator,
    UniformSampler,
)
from opensynth.models.faraday.gaussian_mixture.model import KMeansModel
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    prepare_data_for_model,
)


class KMeansLightningModule(pl.LightningModule):
    """
    Lightning module for training and evaluating a K-Means model.
    """

    def __init__(
        self,
        model: KMeansModel,
        vae_module: pl.LightningModule,
        num_clusters: int,
        num_features: int,
        convergence_tolerance: float = 1e-4,
        predict_target: Literal[
            "assignments", "distances", "inertias"
        ] = "assignments",
    ):
        """
        Args:
            model (KMeansModel): model to train.
            vae_module (pl.LightningModule): VAE module to use for encoding
            the data.
            num_clusters (int): number of clusters in the data
            num_features (int): number of features in the data
            convergence_tolerance (float, optional): training is conducted
                until the Frobenius norm of the change between cluster
                centroids falls below this threshold. Defaults to 1e-4.
            predict_target (Literal[str], optional): whether to predict cluster
                assignments or distances to clusters. Default to "assignments".
        """

        super().__init__()

        self.model = model
        self.vae_module = vae_module
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.convergence_tolerance = convergence_tolerance
        self.predict_target = predict_target

        self.save_hyperparameters(
            "num_clusters",
            "num_features",
            "convergence_tolerance",
            "predict_target",
        )

        # Initialize aggregators
        self.centroid_aggregator = CentroidAggregator(
            num_clusters=self.num_clusters,
            num_features=self.num_features,
        )

        # Initialize metrics
        self.metric_inertia = MeanMetric()

        self.automatic_optimization = False

        # Required parameter to make DDP training work
        self.register_parameter("__ddp_dummy__", nn.Parameter(torch.empty(1)))

    def configure_optimizers(self) -> None:
        return None

    def configure_callbacks(self) -> List[pl.Callback]:
        if self.convergence_tolerance == 0:
            return []
        early_stopping = EarlyStopping(
            "frobenius_norm_change",
            patience=100000,
            stopping_threshold=self.convergence_tolerance,
            check_on_train_epoch_end=True,
            strict=False,  # Allows to not log every epoch
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        self.centroid_aggregator.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # First, we compute the cluster assignments
        encoded_batch = prepare_data_for_model(self.vae_module, batch)
        _, assignments, inertias = self.model.forward(encoded_batch)

        # Then, we update the centroids
        self.centroid_aggregator.update(encoded_batch, assignments)

        # And log the inertia
        self.metric_inertia.update(inertias)
        self.log(
            "inertia",
            self.metric_inertia,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_train_epoch_end(self) -> None:
        centroids = self.centroid_aggregator.compute()

        self.log(
            "frobenius_norm_change",
            torch.linalg.norm(self.model.centroids - centroids),
        )
        self.model.centroids.copy_(centroids)

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        _, _, inertias = self.model.forward(
            prepare_data_for_model(self.vae_module, batch)
        )
        self.metric_inertia.update(inertias)
        self.log("inertia", self.metric_inertia)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        distances, assignments, inertias = self.model.forward(
            prepare_data_for_model(self.vae_module, batch)
        )
        if self.predict_target == "assignments":
            return assignments
        if self.predict_target == "inertias":
            return inertias
        return distances


# -----------------------------------------------------------------------
# INIT STRATEGIES


class KmeansRandomInitLightningModule(pl.LightningModule):
    """
    Lightning module for initializing K-Means centroids randomly.

    Within the first epoch, all items are sampled. Thus, this module should
    only be trained for a single epoch.
    """

    def __init__(
        self,
        model: KMeansModel,
        vae_module: pl.LightningModule,
        num_clusters: int,
        num_features: int,
    ):
        """
        Args:
            model: the model to initialize.
            vae_module: the VAE module to use for encoding the data.
            num_clusters: number of clusters in the data.
            num_features: number of features in the data.
        """
        super().__init__()

        self.model = model
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.vae_module = vae_module

        self.sampler = UniformSampler(
            num_choices=self.num_clusters,
            num_features=self.num_features,
        )

        self.automatic_optimization = False

        # Required parameter to make DDP training work
        self.register_parameter("__ddp_dummy__", nn.Parameter(torch.empty(1)))

    def configure_optimizers(self) -> None:
        return None

    def on_train_epoch_start(self) -> None:
        self.sampler.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.sampler.update(prepare_data_for_model(self.vae_module, batch))

    def on_train_epoch_end(self) -> None:
        choices = self.sampler.compute()
        if choices.dim() > self.model.centroids.dim():
            choices = choices.mean(dim=0)  # Average across all processes
        self.model.centroids.copy_(choices)
