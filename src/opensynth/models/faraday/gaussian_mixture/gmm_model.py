# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import nn

# -------------------------------------------------------------------------------------------------
# GMM Model
# -------------------------------------------------------------------------------------------------


class GaussianMixtureModel(nn.Module):
    """
    PyTorch module for a Gaussian mixture model.

    Covariances are represented via their Cholesky decomposition for
    computational efficiency. The model does not have trainable parameters.
    """

    #: The probabilities of each component,
    # buffer of shape ``[num_components]``.
    component_probs: torch.Tensor
    #: The means of each component,
    # buffer of shape ``[num_components, num_features]``.
    means: torch.Tensor
    #: The precision matrices for the components' covariances,
    # buffer with a shape dependent
    #: on the covariance type, see :class:`CovarianceType`.
    precisions_cholesky: torch.Tensor

    def __init__(self, num_components, num_features):
        """
        Args:
            config: The configuration to use for initializing the
                module's buffers.
        """
        super().__init__()

        self.num_components = num_components
        self.num_features = num_features

        self.register_buffer(
            "component_probs", torch.empty(self.num_components)
        )
        self.register_buffer(
            "means", torch.empty(self.num_components, self.num_features)
        )

        shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )

        self.register_buffer("precisions_cholesky", torch.empty(shape))

        self.reset_parameters()

    @property
    def covariances(self) -> torch.Tensor:
        """
        The covariance matrices learnt for the GMM's components.

        The shape of the tensor is
        num_components x num_features x num_features.
        """
        choleksy_covars = torch.linalg.inv(self.precisions_cholesky)
        return torch.bmm(choleksy_covars.transpose(1, 2), choleksy_covars)

    def reset_parameters(self) -> None:
        """
        Resets the parameters of the GMM.

        - Component probabilities are initialized via uniform sampling and
            normalization.
        - Means are initialized randomly from a Standard Normal.
        - Cholesky precisions are initialized randomly based on the covariance
            type. For all covariance types, it is based on uniform sampling.
        """
        nn.init.uniform_(self.component_probs)
        self.component_probs.div_(self.component_probs.sum())

        nn.init.normal_(self.means)

        nn.init.uniform_(self.precisions_cholesky)
        self.precisions_cholesky.tril_()

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log-probability of observing each of the provided
            datapoints for each of the GMM's components.

        Args:
            data: A tensor of shape ``[num_datapoints, num_features]`` for
                which to compute the log-probabilities.

        Returns:
            - A tensor of shape ``[num_datapoints, num_components]`` with the
                log-responsibilities for each datapoint and components. These
                are the logits of the Categorical distribution over the
                parameters.
            - A tensor of shape ``[num_datapoints]`` with the log-likelihood of
                each datapoint.
        """

        # Precision shape is `[num_components, dim, dim]`.
        log_prob = data.new_empty((data.size(0), self.means.size(0)))
        # We loop here to not blow up the size of intermediate matrices
        for k, (mu, prec_chol) in enumerate(
            zip(self.means, self.precisions_cholesky)
        ):
            inner = data.matmul(prec_chol) - mu.matmul(prec_chol)
            log_prob[:, k] = inner.square().sum(1)

        num_features = data.size(1)

        logdet = (
            self.precisions_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        )

        constant = math.log(2 * math.pi) * num_features

        log_probabilities = logdet - 0.5 * (constant + log_prob)

        log_responsibilities = log_probabilities + self.component_probs.log()
        log_prob = log_responsibilities.logsumexp(1, keepdim=True)
        return log_responsibilities - log_prob, log_prob.squeeze(1)

    def sample(self, num_datapoints: int) -> torch.Tensor:
        """
        Samples the provided number of datapoints from the GMM.

        Args:
            num_datapoints: The number of datapoints to sample.

        Returns:
            A tensor of shape ``[num_datapoints, num_features]`` with the
                random samples.

        Attention:
            This method does not automatically perform batching. If you need to
                sample many
            datapoints, call this method multiple times.
        """
        # First, we sample counts for each

        # Normalise component probabilities after converting to float64
        # This is necessary because numpy casts the probabilities to float64
        # and we want to avoid precision issues resulting from this conversion
        # See https://github.com/numpy/numpy/issues/8317.
        _probs = self.component_probs.numpy().astype("float64")
        component_counts = np.random.multinomial(
            num_datapoints, _probs / _probs.sum()
        )

        # Then, we generate datapoints for each components
        result = []
        for i, count in enumerate(component_counts):
            mean = self.means[i]
            samples = torch.randn(
                count.item(),
                mean.size(0),
                dtype=mean.dtype,
                device=mean.device,
            )

            cholesky_precisions = self.precisions_cholesky[i]
            # For complex covariance types, invert the
            num_features = cholesky_precisions.size(-1)
            target = torch.eye(
                num_features,
                dtype=cholesky_precisions.dtype,
                device=cholesky_precisions.device,
            )
            chol_covariance = torch.linalg.solve_triangular(
                cholesky_precisions, target, upper=True
            ).t()

            scale = chol_covariance.matmul(samples.unsqueeze(-1)).squeeze(-1)

            sample = mean + scale

            result.append(sample)

        return torch.cat(result, dim=0)

    def _get_component_precision(self, component: int) -> torch.Tensor:
        return self.precisions_cholesky[component]
