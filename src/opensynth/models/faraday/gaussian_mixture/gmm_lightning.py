"""
Lightning modules for training and initialising a GMM model.
Code is based on the PyCave framework.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torchmetrics import MeanMetric

from opensynth.models.faraday.gaussian_mixture.metrics import (
    CovarianceAggregator,
    MeanAggregator,
    PriorAggregator,
)
from opensynth.models.faraday.gaussian_mixture.model import (
    GaussianMixtureModel,
)


class GaussianMixtureLightningModule(pl.LightningModule):
    """
    A lightning module which sets some defaults for training models with no
    parameters (i.e. only buffers that are optimized differently than via
    gradient descent).
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        vae_module: pl.LightningModule,
        num_components: int,
        num_features: int,
        covariance_type: str = "full",
        convergence_tolerance: float = 1e-6,
        covariance_regularization: float = 1e-6,
        is_batch_training: bool = False,
    ):
        super().__init__()
        self.model = model
        self.num_components = num_components
        self.num_features = num_features
        self.covariance_type = covariance_type
        self.convergence_tolerance = convergence_tolerance
        self.is_batch_training = is_batch_training
        self.vae_module = vae_module
        self.save_hyperparameters(
            "num_components",
            "num_features",
            "covariance_type",
            "convergence_tolerance",
            "is_batch_training",
        )

        # For batch training, we store a model copy such that we can "replay"
        # responsibilities
        if self.is_batch_training:
            self.model_copy = GaussianMixtureModel(
                self.covariance_type, self.num_components, self.num_features
            )
            self.model_copy.load_state_dict(self.model.state_dict())

        # Initialize aggregators
        self.prior_aggregator = PriorAggregator(
            num_components=self.num_components,
        )
        self.mean_aggregator = MeanAggregator(
            num_components=self.num_components,
            num_features=self.num_features,
        )
        self.covar_aggregator = CovarianceAggregator(
            num_components=self.num_components,
            num_features=self.num_features,
            covariance_type=self.covariance_type,
            reg=covariance_regularization,
        )
        # Initialize metrics
        self.metric_nll = MeanMetric()

        self.automatic_optimization = False

        # Required parameter to make DDP training work
        self.register_parameter("__ddp_dummy__", nn.Parameter(torch.empty(1)))

    def configure_optimizers(self) -> None:
        return None

    def configure_callbacks(self) -> list[pl.Callback]:
        if self.convergence_tolerance == 0:
            return []
        early_stopping = EarlyStopping(
            "nll",
            min_delta=self.convergence_tolerance,
            patience=2 if self.is_batch_training else 1,
            check_on_train_epoch_end=True,
            strict=False,  # Allows to not log every epoch
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.mean_aggregator.reset()
        self.covar_aggregator.reset()

    def training_step(self, batch: torch.Tensor) -> None:

        # Encode the batch
        encoded_batch = self.prepare_data_for_model(batch)

        if self._computes_responsibilities_on_live_model:
            log_responsibilities, log_probs = self.model.forward(encoded_batch)
        else:
            log_responsibilities, log_probs = self.model_copy.forward(
                encoded_batch
            )
        responsibilities = log_responsibilities.exp()

        # Compute the NLL for early stopping
        if self._should_log_nll:
            self.metric_nll.update(-log_probs)

        if self._should_update_means:
            self.prior_aggregator.update(responsibilities)
            self.mean_aggregator.update(encoded_batch, responsibilities)
            if self._should_update_covars:
                means = self.mean_aggregator.compute()
                self.covar_aggregator.update(
                    encoded_batch, responsibilities, means
                )
        else:
            self.covar_aggregator.update(
                encoded_batch,
                responsibilities,
                self.model.means,
            )

    def on_train_epoch_end(self) -> None:
        # Prior to updating the model, we might need to copy it in the case of
        # batch training
        if self._requires_to_copy_live_model:
            self.model_copy.load_state_dict(self.model.state_dict())

        if self._should_log_nll:
            self.log(
                "nll",
                self.metric_nll,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        # Finalize the M-Step
        if self._should_update_means:
            priors = self.prior_aggregator.compute()
            self.model.component_probs.copy_(priors)

            means = self.mean_aggregator.compute()
            self.model.means.copy_(means)

        if self._should_update_covars:
            covars = self.covar_aggregator.compute()
            self.model.precisions_cholesky.copy_(
                cholesky_precision(covars, self.covariance_type)
            )

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        _, log_probs = self.model.forward(self.prepare_data_for_model(batch))
        self.metric_nll.update(-log_probs)
        self.log("nll", self.metric_nll)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_responsibilities, log_probs = self.model.forward(
            self.prepare_data_for_model(batch)
        )
        return log_responsibilities.exp(), -log_probs

    @property
    def _computes_responsibilities_on_live_model(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 0

    @property
    def _requires_to_copy_live_model(self) -> bool:
        if not self.is_batch_training:
            return False
        return self.current_epoch % 2 == 0

    @property
    def _should_log_nll(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 1

    @property
    def _should_update_means(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 0

    @property
    def _should_update_covars(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 1

    def prepare_data_for_model(self, batch: torch.Tensor):
        """Prepare data for the model by encoding it with the VAE and adding
            month and day of week.

        Args:
            batch (torch.Tensor): a batch of data to prepare.
        Returns:
            torch.Tensor: model inputs consisting of encoded data, month,
            and day of week.
        """
        kwh = batch["kwh"]
        features = batch["features"]
        vae_input = self.vae_module.reshape_data(kwh, features)
        vae_output = self.vae_module.encode(vae_input)
        gmm_input = self.vae_module.reshape_data(vae_output, features)

        return gmm_input


class GaussianMixtureInitLightningModule(pl.LightningModule):
    """
    Lightning module for initializing a Gaussian mixture from centroids found
    via K-Means.
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        vae_module: pl.LightningModule,
        num_components: int,
        num_features: int,
        init_method: str = "kmeans",
        covariance_type: str = "full",
        covariance_regularization: float = 1e-6,
        is_batch_training: bool = True,
    ):
        """
        Args:
            model: The model whose parameters to initialize.
            vae_module: The VAE module to use for encoding the data.
            covariance_regularization: A small value which is added to the
            diagonal of the covariance matrix to ensure that it is positive
            semi-definite.
        """
        super().__init__()

        self.model = model
        self.num_components = num_components
        self.num_features = num_features
        self.init_method = init_method
        self.covariance_type = covariance_type
        self.is_batch_training = is_batch_training
        self.vae_module = vae_module
        self.save_hyperparameters("init_method")

        self.prior_aggregator = PriorAggregator(
            num_components=self.num_components,
        )

        self.covar_aggregator = CovarianceAggregator(
            num_components=self.num_components,
            num_features=self.num_features,
            covariance_type=self.covariance_type,
            reg=covariance_regularization,
        )

        if self.init_method == "rand":
            self.mean_aggregator = MeanAggregator(
                num_components=self.num_components,
                num_features=self.num_features,
            )
            if self.is_batch_training:
                # For batch training, we store a model copy such that we can
                # "replay" responsibilities
                self.model_copy = GaussianMixtureModel(
                    self.covariance_type,
                    self.num_components,
                    self.num_features,
                )
                self.model_copy.load_state_dict(self.model.state_dict())

        self.automatic_optimization = False

        # Required parameter to make DDP training work
        self.register_parameter("__ddp_dummy__", nn.Parameter(torch.empty(1)))

    def configure_optimizers(self) -> None:
        return None

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.covar_aggregator.reset()
        if self.init_method == "rand":
            self.mean_aggregator.reset()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:

        # Encode the batch
        encoded_batch = self.prepare_data_for_model(batch)

        if self.init_method == "kmeans":
            # Just like for k-means, responsibilities are one-hot assignments
            # to the clusters
            responsibilities = self._one_hot_responsibilities(
                encoded_batch, self.model.means
            )

            # Then, we can update the aggregators
            self.prior_aggregator.update(responsibilities)
            self.covar_aggregator.update(
                encoded_batch,
                responsibilities,
                self.model.means,
            )

        elif self.init_method == "rand":
            means = self.mean_aggregator.compute()

            responsibilities = torch.rand(
                encoded_batch.size(0),
                self.num_components,
                device=encoded_batch.device,
                dtype=self.encoded_batch.dtype,
            )
            responsibilities = responsibilities / responsibilities.sum(
                1, keepdim=True
            )

            if self.current_epoch == 0:
                self.prior_aggregator.update(responsibilities)
                means = self.mean_aggregator.compute()
                self.mean_aggregator.update(encoded_batch, responsibilities)
                if not self.is_batch_training:
                    means = self.mean_aggregator.compute()
                    self.covar_aggregator.update(
                        encoded_batch, responsibilities, means
                    )
            else:
                # Only reached if batch training
                self.covar_aggregator.update(
                    encoded_batch, responsibilities, self.model.means
                )

        else:
            raise Warning(
                "Invalid initialization method. Choose 'kmeans' or 'rand'."
            )

    def on_train_epoch_end(self) -> None:
        if self.init_method == "kmeans":
            priors = self.prior_aggregator.compute()
            self.model.component_probs.copy_(priors)

            covars = self.covar_aggregator.compute()
            self.model.precisions_cholesky.copy_(
                cholesky_precision(covars, self.covariance_type)
            )

        elif self.init_method == "rand":
            if self.current_epoch == 0 and self.is_batch_training:
                self.model_copy.load_state_dict(self.model.state_dict())

            if self.current_epoch == 0:
                priors = self.prior_aggregator.compute()
                self.model.component_probs.copy_(priors)

                means = self.mean_aggregator.compute()
                self.model.means.copy_(means)

            if (
                self.current_epoch == 0 and not self.is_batch_training
            ) or self.current_epoch == 1:
                covars = self.covar_aggregator.compute()
                self.model.precisions_cholesky.copy_(
                    cholesky_precision(covars, self.covariance_type)
                )

    def _one_hot_responsibilities(
        self, data: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        distances = torch.cdist(data, centroids)
        assignments = distances.min(1).indices
        onehot = torch.eye(
            centroids.size(0),
            device=data.device,
            dtype=data.dtype,
        )
        return onehot[assignments]

    def prepare_data_for_model(self, batch: torch.Tensor):
        """Prepare data for the model by encoding it with the VAE and adding
            month and day of week.

        Args:
            batch (torch.Tensor): a batch of data to prepare.
        Returns:
            torch.Tensor: model inputs consisting of encoded data, month,
                and day of week.
        """
        kwh = batch["kwh"]
        features = batch["features"]
        vae_input = self.vae_module.reshape_data(kwh, features)
        vae_output = self.vae_module.encode(vae_input)
        gmm_input = self.vae_module.reshape_data(vae_output, features)

        return gmm_input


def cholesky_precision(
    covariances: torch.Tensor, covariance_type: str
) -> torch.Tensor:
    """
    Computes the Cholesky decompositions of the precision matrices induced by
        the provided covariance matrices.

    Args:
        covariances: A tensor of shape
            ``[num_components, dim, dim]``, ``[dim, dim]``,
            ``[num_components, dim]``, ``[dim]`` or
            ``[num_components]`` depending on the
            ``covariance_type``. These are the covariance matrices of
            multivariate Normal distributions.
        covariance_type: The type of covariance for the covariance matrices
            given.

    Returns:
        A tensor of the same shape as ``covariances``, providing the
        lower-triangular Cholesky decompositions of the precision matrices.
    """
    if covariance_type in ("tied", "full"):
        # Compute Cholesky decomposition
        cholesky = torch.linalg.cholesky(covariances)
        # Invert
        num_features = covariances.size(-1)
        target = torch.eye(
            num_features, dtype=covariances.dtype, device=covariances.device
        )
        if covariance_type == "full":
            num_components = covariances.size(0)
            target = target.unsqueeze(0).expand(num_components, -1, -1)
        return torch.linalg.solve_triangular(
            cholesky, target, upper=False
        ).transpose(-2, -1)

    # "Simple" kind of covariance
    return covariances.sqrt().reciprocal()
