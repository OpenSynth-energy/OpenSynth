# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric


class WeightsMetric(Metric):
    full_state_update = False

    def __init__(self, num_components: int):
        super().__init__()
        self.weights: torch.Tensor

        self.add_state(
            "weights", torch.zeros(num_components), dist_reduce_fx="mean"
        )

    def update(self, weights: torch.Tensor) -> None:
        self.weights.add_(weights)

    def compute(self) -> None:
        return self.weights


class MeansMetric(Metric):
    full_state_update = False

    def __init__(self, num_components: int, num_features: int):
        super().__init__()
        self.means: torch.Tensor
        self.add_state(
            "means",
            torch.zeros(num_components, num_features),
            dist_reduce_fx="mean",
        )

    def update(self, means: torch.Tensor) -> None:
        self.means.add_(means)

    def compute(self) -> None:
        return self.means


class PrecisionCholeskyMetric(Metric):
    full_state_update = False

    def __init__(self, num_components: int, num_features: int):
        super().__init__()
        self.precision_cholesky: torch.Tensor
        self.add_state(
            "precision_cholesky",
            torch.zeros(num_components, num_features, num_features),
            dist_reduce_fx="mean",
        )

    def update(self, precision_cholesky: torch.Tensor) -> None:
        self.precision_cholesky.add_(precision_cholesky)

    def compute(self) -> None:
        return self.precision_cholesky


class CovarianceMetric(Metric):
    full_state_update = False

    def __init__(self, num_components: int, num_features: int):
        super().__init__()
        self.covariances: torch.Tensor
        self.add_state(
            "covariances",
            torch.zeros(num_components, num_features, num_features),
            dist_reduce_fx="mean",
        )

    def update(self, covariances: torch.Tensor) -> None:
        self.covariances.add_(covariances)

    def compute(self) -> None:
        return self.covariances


class NegativeLogLikelihoodMetric(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.nll: torch.Tensor
        self.add_state(
            "nll",
            torch.zeros(1),
            dist_reduce_fx="mean",
        )

    def update(self, nll: torch.Tensor) -> None:
        self.nll.add_(nll)

    def compute(self) -> None:
        return self.nll
