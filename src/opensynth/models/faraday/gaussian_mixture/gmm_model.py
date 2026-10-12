# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, TypedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

from opensynth.models.faraday.gaussian_mixture import gmm_metrics, gmm_utils
from opensynth.models.faraday.vae_model import FaradayVAE


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
    nll: torch.Tensor

    def __init__(
        self,
        num_components: int,
        num_features: int,
        reg_covar: float = 1e-6,
    ):

        super().__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.reg_covar = reg_covar

        # Initialise model params
        weights_shape = torch.Size([self.num_components])
        means_shape = torch.Size([self.num_components, self.num_features])
        precision_cholesky_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        covariances_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        nll_shape = torch.Size([1])
        self.register_buffer("weights", torch.empty(weights_shape))
        self.register_buffer("means", torch.empty(means_shape))
        self.register_buffer(
            "precision_cholesky", torch.empty(precision_cholesky_shape)
        )
        self.register_buffer("covariances", torch.empty(covariances_shape))
        self.register_buffer("nll", torch.empty(nll_shape))
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
            torch.Tensor: Log probability
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

    def _estimate_log_weights(self) -> torch.Tensor:
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

    def m_step(
        self, X: torch.Tensor, log_responsibilities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute this batch's contribution to the M-step sufficient
        statistics. These are additive across batches: the caller is
        expected to accumulate them across every batch in an epoch and
        pass the totals to `update_params_from_statistics` once, rather
        than updating the model parameters after each batch.

        Args:
            X (torch.Tensor): Input data
            log_responsibilities (torch.Tensor): Log responsibilities
                from the e-step

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: this batch's
            nk, sk and sk2 sufficient statistics. See
            `gmm_utils.torch_compute_sufficient_statistics`.
        """
        return gmm_utils.torch_compute_sufficient_statistics(
            X, responsibilities=torch.exp(log_responsibilities)
        )

    def update_params_from_statistics(
        self,
        nk: torch.Tensor,
        sk: torch.Tensor,
        sk2: torch.Tensor,
        nll: torch.Tensor,
    ):
        """
        Perform the M-step update from sufficient statistics accumulated
        across all batches in an epoch, and update the model parameters.

        Args:
            nk (torch.Tensor): Accumulated sum of responsibilities per
                component
            sk (torch.Tensor): Accumulated responsibility-weighted sum of
                X per component
            sk2 (torch.Tensor): Accumulated responsibility-weighted sum of
                outer products of X per component
            nll (torch.Tensor): Negative log likelihood to record
        """
        weights_, means_, covariances_ = (
            gmm_utils.torch_estimate_gaussian_parameters_from_statistics(
                nk, sk, sk2, reg_covar=self.reg_covar
            )
        )
        precision_cholesky_ = gmm_utils.torch_compute_precision_cholesky(
            covariances=covariances_, reg=self.reg_covar
        )
        return self.update_params(
            weights=weights_,
            means=means_,
            precision_cholesky=precision_cholesky_,
            covariances=covariances_,
            nll=nll,
        )

    def update_params(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        precision_cholesky: torch.Tensor,
        covariances: torch.Tensor,
        nll: torch.Tensor,
    ):
        self.weights.data = weights
        self.means.data = means
        self.precision_cholesky.data = precision_cholesky
        self.covariances.data = covariances
        self.nll.data = nll
        return self

    def forward(self, X: torch.Tensor):
        return self.e_step(X)

    def predict(self, X: torch.Tensor):
        return self._estimate_weighted_log_prob(X).argmax(dim=1)

    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from GMM components

        Args:
            n_samples (int): number of samples to generate

        Returns:
            torch.Tensor: samples drawn from GMM size n_samples x n_components
        """

        # Set up the random generator
        generator = torch.Generator()
        # Sample component counts from the multinomial distribution
        n_samples_comp = torch.multinomial(
            self.weights, n_samples, replacement=True, generator=generator
        ).bincount(minlength=len(self.weights))

        # Initialize list to collect samples
        X = []

        # Sample from each component based on the number of samples
        for mean, covariance, sample_count in zip(
            self.means, self.covariances, n_samples_comp
        ):
            if (
                sample_count > 0
            ):  # Only sample if we need samples from this component
                dist = torch.distributions.MultivariateNormal(mean, covariance)
                samples = dist.sample((sample_count,))
                X.append(samples)

        return torch.vstack(X)


class GaussianMixtureLightningModule(pl.LightningModule):

    def __init__(
        self,
        gmm_module: GaussianMixtureModel,
        vae_module: FaradayVAE,
        num_components: int,
        num_features: int,
        reg_covar: float = 1e-6,
        convergence_tolerance: float = 1e-2,
        sample_weights_column: Optional[str] = None,
    ):
        super().__init__()
        self.gmm_module = gmm_module
        self.vae_module = vae_module
        self.num_components = num_components
        self.num_features = num_features
        self.reg_covar = reg_covar

        self.automatic_optimization = False
        self.convergence_tolerance = convergence_tolerance

        # M-step sufficient statistics, accumulated across every batch in
        # an epoch (and across devices, in distributed training). The
        # mixture parameters are updated once per epoch from these totals,
        # rather than being overwritten by each batch's M-step in
        # isolation.
        self.nk_metric = gmm_metrics.SufficientStatisticMetric(
            torch.Size([self.num_components])
        )
        self.sk_metric = gmm_metrics.SufficientStatisticMetric(
            torch.Size([self.num_components, self.num_features])
        )
        self.sk2_metric = gmm_metrics.SufficientStatisticMetric(
            torch.Size(
                [self.num_components, self.num_features, self.num_features]
            )
        )
        self.nll = gmm_metrics.NegativeLogLikelihoodMetric()

        self.sample_weights_column = sample_weights_column

    def configure_optimizers(self) -> None:
        return None

    def on_train_epoch_start(self) -> None:
        # At the start of epoch, reset the accumulated statistics
        self.nk_metric.reset()
        self.sk_metric.reset()
        self.sk2_metric.reset()
        self.nll.reset()

    def training_step(self, batch) -> None:
        # Encode the batch
        encoded_batch = gmm_utils.prepare_data_for_training_step(
            batch, self.vae_module, self.sample_weights_column
        )

        # Run e-step using the mixture parameters from the previous
        # epoch's update. Parameters stay fixed for the whole epoch so
        # that every batch's statistics are computed against the same
        # model.
        log_prob, log_resp = self.gmm_module.e_step(encoded_batch)

        # Accumulate this batch's contribution to the M-step sufficient
        # statistics, instead of updating the mixture parameters
        # directly from a single batch.
        nk, sk, sk2 = self.gmm_module.m_step(encoded_batch, log_resp)
        self.nk_metric.update(nk)
        self.sk_metric.update(sk)
        self.sk2_metric.update(sk2)
        self.nll.update(torch.neg(log_prob))

    def on_train_epoch_end(self) -> None:
        # Combine the statistics accumulated across every batch (and, in
        # distributed training, every device) into a single M-step update.
        nk = self.nk_metric.compute()
        sk = self.sk_metric.compute()
        sk2 = self.sk2_metric.compute()
        nll = self.nll.compute()

        self.log(
            "nll",
            nll,
            on_step=False,
            on_epoch=True,
        )

        self.gmm_module.update_params_from_statistics(
            nk=nk, sk=sk, sk2=sk2, nll=nll
        )

    def configure_callbacks(self) -> list[pl.Callback]:
        early_stopping = EarlyStopping(
            "nll",
            min_delta=self.convergence_tolerance,
            patience=1,
            mode="min",
        )
        return [early_stopping]
