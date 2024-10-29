from typing import Tuple, TypedDict

import torch
import torch.nn as nn

from opensynth.models.faraday.new_gmm import gmm_utils


class GMMInitParams(TypedDict):
    labels: torch.Tensor
    means: torch.Tensor
    responsibilities: torch.Tensor
    weights: torch.Tensor
    covariances: torch.Tensor
    precision_cholesky: torch.Tensor
    components_probability: torch.Tensor


class GaussianMixtureModel(nn.Module):

    components_probability: torch.Tensor
    means: torch.Tensor
    precisions_cholesky: torch.Tensor

    def __init__(self, num_components: int, num_features: int):

        super().__init__()
        self.num_components = num_components
        self.num_features = num_features

        self.register_buffer(
            "component_probs", torch.empty(self.num_components)
        )
        self.register_buffer(
            "means", torch.empty(self.num_components, self.num_features)
        )

        precision_cholesky_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )
        self.register_buffer(
            "precisions_cholesky", torch.empty(precision_cholesky_shape)
        )

        self.means = None
        self.precision_cholesky = None
        self.components_probability = None

    def initialise(self, init_params: GMMInitParams):
        self.means = init_params["means"]
        self.precision_cholesky = init_params["precision_cholesky"]
        self.components_probability = init_params["components_probability"]

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
        if self.precision_cholesky is None or self.means is None:
            raise AttributeError("Model is not initialised.")

        n_samples, n_features = X.shape
        # Log determinant of cholesky matrix
        log_det = self._compute_log_det_cholesky(
            self.precision_cholesky, n_features
        )
        # Log of probabilities
        log_prob = torch.empty((n_samples, self.num_components))
        for k, (mu, prec_chol) in enumerate(
            zip(self.means, self.precision_cholesky)
        ):
            y = torch.matmul(X, prec_chol) - torch.matmul(mu, prec_chol)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        # log gaussian likelihood
        return (
            -0.5
            * (n_features * torch.log(2 * torch.tensor(torch.pi)) + log_prob)
            + log_det
        )

    def _estimate_log_weights(self: torch.Tensor) -> torch.Tensor:
        """
        Estimate log of component_proba.
        Pytorch implementation of sklearns's
        sklearn.mixture._base.BaseMixture._estimate_log_weights

        Returns:
            torch.Tensor: Log of weights
        """
        if self.components_probability is None:
            raise AttributeError("Model is not initialised.")

        return torch.log(self.components_probability)

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
                means=self.means,
                responsibilities=torch.exp(log_reponsibilities),
                reg_covar=1e-6,
            )
        )

        precision_cholesky = gmm_utils.torch_compute_precision_cholesky(
            covariances_=covariances_
        )

        # Update state
        self.precisions_cholesky = precision_cholesky
        self.weights_ = weights_
        self.means = means_

    def forward(self):
        pass
