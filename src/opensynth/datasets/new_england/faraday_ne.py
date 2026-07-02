# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Faraday model with conditional GMM sampling for New England.

FaradayModel fits its GMM over the joint space of latent codes and
conditioning labels, then samples unconditionally. Dataset
generation needs the opposite: draw load profiles for *chosen*
labels (state, archetype, month, ...). For a Gaussian mixture the
conditional distribution z | y = y* is available in closed form per
component, with component weights reweighted by each component's
likelihood of y*.
"""

import logging

import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.model import FaradayModel

logger = logging.getLogger(__name__)

_JITTER = 1e-6
_MAX_JITTER_TRIES = 5


def _stable_cholesky(mat: torch.Tensor) -> torch.Tensor:
    """
    Batched Cholesky with escalating diagonal jitter.

    Args:
        mat (torch.Tensor): Symmetric matrices [..., D, D].

    Returns:
        torch.Tensor: Lower-triangular factors.
    """
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype)
    jitter = _JITTER
    for _ in range(_MAX_JITTER_TRIES):
        try:
            return torch.linalg.cholesky(mat + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10
    raise torch.linalg.LinAlgError(
        f"Cholesky failed with jitter up to {jitter:.0e}"
    )


class NewEnglandFaradayModel(FaradayModel):
    """
    FaradayModel with closed-form conditional sampling.
    """

    def _label_vector(self, labels: dict[str, float]) -> torch.Tensor:
        """
        Order a label dict into the GMM's feature vector layout.

        Args:
            labels (dict[str, float]): One value per feature in
                self.feature_list.

        Returns:
            torch.Tensor: Label vector [n_labels].
        """
        if set(labels) != set(self.feature_list):
            raise ValueError(
                f"Labels {sorted(labels)} do not match the model's "
                f"feature list {sorted(self.feature_list)}"
            )
        return torch.tensor(
            [float(labels[f]) for f in self.feature_list],
            dtype=torch.float64,
        )

    def _conditional_mixture(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Condition every GMM component on the label vector y.

        Args:
            y (torch.Tensor): Label vector [n_labels].

        Returns:
            tuple: (component probabilities [K], conditional means
            [K, latent_dim], conditional covariance Cholesky factors
            [K, latent_dim, latent_dim]).
        """
        latent = self.vae_module.latent_dim
        means = self.gmm_module.means.double()
        covs = self.gmm_module.covariances.double()
        weights = self.gmm_module.weights.double()

        mu_z, mu_y = means[:, :latent], means[:, latent:]
        s_zz = covs[:, :latent, :latent]
        s_zy = covs[:, :latent, latent:]
        s_yy = covs[:, latent:, latent:]
        eye_y = torch.eye(s_yy.shape[-1], dtype=s_yy.dtype)
        s_yy = s_yy + _JITTER * eye_y

        diff = (y - mu_y).unsqueeze(-1)  # [K, n_labels, 1]
        solved = torch.linalg.solve(s_yy, diff)
        cond_means = mu_z + (s_zy @ solved).squeeze(-1)
        cond_covs = s_zz - s_zy @ torch.linalg.solve(
            s_yy, s_zy.transpose(1, 2)
        )
        # Symmetrise against numerical drift before factorising
        cond_covs = 0.5 * (cond_covs + cond_covs.transpose(1, 2))
        cond_chol = _stable_cholesky(cond_covs)

        # Reweight components by their likelihood of y
        mvn_y = torch.distributions.MultivariateNormal(
            mu_y, covariance_matrix=s_yy
        )
        log_w = torch.log(weights + 1e-300) + mvn_y.log_prob(y)
        probs = torch.softmax(log_w, dim=0)
        return probs, cond_means, cond_chol

    def sample_gmm_conditional(
        self, labels: dict[str, float], n_samples: int
    ) -> TrainingData:
        """
        Sample synthetic profiles for fixed conditioning labels.

        Args:
            labels (dict[str, float]): One integer-encoded value per
                feature in self.feature_list.
            n_samples (int): Number of profiles to generate.

        Returns:
            TrainingData: Decoded kWh and the (constant) labels.
        """
        y = self._label_vector(labels)
        probs, cond_means, cond_chol = self._conditional_mixture(y)

        components = torch.multinomial(probs, n_samples, replacement=True)
        noise = torch.randn(
            n_samples, cond_means.shape[-1], dtype=torch.float64
        )
        z = cond_means[components] + (
            cond_chol[components] @ noise.unsqueeze(-1)
        ).squeeze(-1)

        features = {
            f: torch.full((n_samples, 1), float(labels[f]))
            for f in self.feature_list
        }
        decoder_input = self.vae_module.reshape_data(
            z.float(), features
        ).float()
        kwh = self.vae_module.decode(decoder_input)
        return TrainingData(kwh=kwh, features=features)
