from typing import Tuple, TypedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.new_gmm import gmm_metrics, gmm_utils


class GMMInitParams(TypedDict):
    labels: torch.Tensor
    means: torch.Tensor
    responsibilities: torch.Tensor
    weights: torch.Tensor
    covariances: torch.Tensor
    precision_cholesky: torch.Tensor


class GaussianMixtureModel(nn.Module):

    weights: torch.Tensor
    means: torch.Tensor
    precision_cholesky: torch.Tensor
    covariances: torch.Tensor
    log_prob: torch.Tensor

    def __init__(
        self,
        num_components: int,
        num_features: int,
        reg_covar: float = 1e-6,
        print_idx: int = 0,
    ):

        super().__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.reg_covar = reg_covar
        self.print_idx = print_idx

        # Initialise model params
        weights_shape = torch.Size([self.num_components])
        means_shape = torch.Size([self.num_components, self.num_features])
        precision_cholesky_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        covariances_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        log_prob_shape = torch.Size([1])
        self.register_buffer("weights", torch.empty(weights_shape))
        self.register_buffer("means", torch.empty(means_shape))
        self.register_buffer(
            "precision_cholesky", torch.empty(precision_cholesky_shape)
        )
        self.register_buffer("covariances", torch.empty(covariances_shape))
        self.register_buffer("log_prob", torch.empty(log_prob_shape))
        self.initialised = False

    def initialise(self, init_params: GMMInitParams):
        self.means.data = init_params["means"]
        self.precision_cholesky.data = init_params["precision_cholesky"]
        self.weights.data = init_params["weights"]
        self.initialised = True

    @staticmethod
    def _compute_log_det_cholesky(
        matrix_chol: torch.Tensor, n_features: int
    ) -> torch.Tensor:
        """
        Compute the log-det of the cholesky decomposition of matrices.
        Pytorch implementation of sklearn's
        sklearn.mixture._gaussian_mixture._compute_log_det_cholesky

        Args:
            matrix_chol (torch.Tensor): Cholesky matrix
            n_features (int): Number of features

        Returns:
            torch.Tensor: Log determinant of cholesky matrix
        """
        n_components, _, _ = matrix_chol.shape
        log_det_chol = torch.sum(
            torch.log(
                matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]
            ),
            dim=1,
        )
        log_det_chol = log_det_chol.to(matrix_chol.device)
        return log_det_chol

    def _estimate_log_gaussian_prob(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the log gaussian probability.
        Pytorch implementation of sklearn's
        sklearn.mixture._gaussian_mixture._estimate_log_gaussian_prob

        Args:
            X (torch.Tensor): Input data

        Returns:
            torch.Tensor: Log probabability
        """
        if self.initialised is False:
            raise AttributeError("Model is not initialised.")
        n_samples, n_features = X.shape
        # Log determinant of cholesky matrix
        log_det = self._compute_log_det_cholesky(
            self.precision_cholesky, n_features
        )
        # Log of probabilities
        log_prob = torch.empty(
            (n_samples, self.num_components), device=X.device
        )
        for k, (mu, prec_chol) in enumerate(
            zip(self.means, self.precision_cholesky)
        ):
            y = torch.matmul(X, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        # log gaussian likelihood

        pi = torch.tensor(torch.pi, device=log_prob.device)
        return -0.5 * (n_features * torch.log(2 * pi) + log_prob) + log_det

    def _estimate_log_weights(self: torch.Tensor) -> torch.Tensor:
        """
        Estimate log of weights.
        Pytorch implementation of sklearns's
        sklearn.mixture._base.BaseMixture._estimate_log_weights

        Returns:
            torch.Tensor: Log of weights
        """
        if self.initialised is False:
            raise AttributeError("Model is not initialised.")

        return torch.log(self.weights)

    def _estimate_weighted_log_prob(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimated weighted log probability.
        Pytorch's implementation of sklearn's
        sklearn.mixture._base.BaseMixture._estimate_weighted_log_prob

        Args:
            X (torch.Tensor): Input data

        Returns:
            torch.Tensor: Weighted log probability
        """
        w = self._estimate_log_weights()
        p = self._estimate_log_gaussian_prob(X)
        return p + w

    def _estimate_log_prob_and_responsibilities(
        self,
        X: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Pytorch implementation of sklearn's
        sklearn.mixture._base.BaseMixture._estimate_log_prob_resp
        Args:
            X (torch.tensor): Input data

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Normalised log probabilities and log responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm.reshape(-1, 1)
        return log_prob_norm, log_resp

    def e_step(
        self,
        X,
    ):
        log_prob_norm, log_resp = self._estimate_log_prob_and_responsibilities(
            X
        )
        return torch.mean(log_prob_norm), log_resp

    def m_step(self, X: torch.Tensor, log_reponsibilities: torch.Tensor):
        weights_, means_, covariances_ = (
            gmm_utils.torch_estimate_gaussian_parameters(
                X,
                responsibilities=torch.exp(log_reponsibilities),
                reg_covar=self.reg_covar,
            )
        )
        precision_cholesky_ = gmm_utils.torch_compute_precision_cholesky(
            covariances=covariances_, reg=self.reg_covar
        )
        return precision_cholesky_, weights_, means_, covariances_

    def update_params(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        precision_cholesky: torch.Tensor,
        covariances: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        self.weights.data = weights
        self.means.data = means
        self.precision_cholesky.data = precision_cholesky
        self.covariances = covariances
        self.log_prob = log_prob
        return self

    def forward(self, X: torch.Tensor):
        return self.e_step(X)

    def predict(self, X: torch.Tensor):
        return self._estimate_weighted_log_prob(X).argmax(dim=1)


class GaussianMixtureLightningModule(pl.LightningModule):

    def __init__(
        self,
        gmm_module: GaussianMixtureModel,
        vae_module: FaradayVAE,
        num_components: int,
        num_features: int,
        reg_covar: float = 1e-6,
        convergence_tolerance: float = 1e-2,
        compute_on_batch: bool = False,
    ):
        super().__init__()
        self.gmm_module = gmm_module
        self.vae_module = vae_module
        self.num_components = num_components
        self.num_features = num_features
        self.reg_covar = reg_covar

        self.automatic_optimization = False
        self.convergence_tolerance = convergence_tolerance
        # self.register_parameter("__ddp_dummy__",
        #  nn.Parameter(torch.empty(1)))

        # GMM params to sync across processes
        self.weight_metric = gmm_metrics.WeightsMetric(self.num_components)
        self.mean_metric = gmm_metrics.MeansMetric(
            self.num_components, self.num_features
        )
        self.precision_cholesky_metric = gmm_metrics.PrecisionCholeskyMetric(
            self.num_components, self.num_features
        )
        self.covariance_metric = gmm_metrics.CovarianceMetric(
            self.num_components, self.num_features
        )
        self.log_prob = gmm_metrics.LogProbMetric()

        self.compute_on_batch = compute_on_batch

    def configure_optimizers(self) -> None:
        return None

    def on_train_epoch_start(self) -> None:
        # At the start of epoch, reset metrics
        self.mean_metric.reset()
        self.weight_metric.reset()
        self.precision_cholesky_metric.reset()
        self.covariance_metric.reset()
        self.log_prob.reset()

    def training_step(self, batch) -> None:

        encoded_batch = batch

        # Run e-step
        log_prob, log_resp = self.gmm_module.e_step(encoded_batch)
        # Run m-step
        precision_cholesky, weights, means, covariances = (
            self.gmm_module.m_step(encoded_batch, log_resp)
        )
        # Update model params. This only updates the params
        # on the current device
        self.gmm_module.update_params(
            weights=weights,
            means=means,
            precision_cholesky=precision_cholesky,
            covariances=covariances,
            log_prob=log_prob,
        )

        if self.compute_on_batch:
            # forward performs update, compute and reset metrics
            weights_reduced = self.weight_metric.forward(weights)
            means_reduced = self.mean_metric.forward(means)
            prec_chol_reduced = self.precision_cholesky_metric.forward(
                precision_cholesky
            )
            covar_reduced = self.covariance_metric.forward(covariances)
            log_prob_reduced = self.log_prob.forward(log_prob)

            self.gmm_module.update_params(
                weights=weights_reduced,
                means=means_reduced,
                precision_cholesky=prec_chol_reduced,
                covariances=covar_reduced,
                log_prob=log_prob_reduced,
            )

    def on_train_epoch_end(self) -> None:
        # At the end of epoch, update metrics and sync across
        # multiple devices using torchmetrics.Metric.compute
        # Then update model params using the synced values

        if not self.compute_on_batch:
            weights = self.gmm_module.weights
            means = self.gmm_module.means
            precision_cholesky = self.gmm_module.precision_cholesky
            covariances = self.gmm_module.covariances
            log_prob = self.gmm_module.log_prob

            self.weight_metric.update(weights)
            self.mean_metric.update(means)
            self.precision_cholesky_metric.update(precision_cholesky)
            self.covariance_metric.update(covariances)
            self.log_prob.update(log_prob)

        weights_reduced = self.weight_metric.compute()
        means_reduced = self.mean_metric.compute()
        prec_chol_reduced = self.precision_cholesky_metric.compute()
        covar_reduced = self.covariance_metric.compute()
        log_prob_reduced = self.log_prob.compute()

        print(
            f"Local weights at rank: {self.local_rank} -",
            f"means: {weights_reduced[0]:.4f}, {means_reduced[0][0]:.4f}",
        )

        if self.local_rank == 0:
            print(
                f"Reduced weights, means: {weights_reduced[0]:.4f}, "
                f"{means_reduced[0][0]:.4f}"
            )

        self.log(
            "abs_log_prob",
            log_prob_reduced,
            on_step=False,
            on_epoch=True,
        )  # uses mean-reduction (default) to accumulate the metrics

        self.gmm_module.update_params(
            weights=weights_reduced,
            means=means_reduced,
            precision_cholesky=prec_chol_reduced,
            covariances=covar_reduced,
            log_prob=log_prob_reduced,
        )

    def configure_callbacks(self) -> list[pl.Callback]:
        early_stopping = EarlyStopping(
            "abs_log_prob",
            min_delta=self.convergence_tolerance,
            patience=1,
            check_on_train_epoch_end=True,
        )
        return [early_stopping]
