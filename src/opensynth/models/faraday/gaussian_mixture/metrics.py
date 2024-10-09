# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import torch
from torchmetrics import Metric


# -----------------------------------------------------------------------------
# GMM Metrics
# -----------------------------------------------------------------------------
class PriorAggregator(Metric):
    """
    The prior aggregator aggregates component probabilities over batches and
        process.
    """

    full_state_update = False

    def __init__(
        self,
        num_components: int,
    ):
        super().__init__()

        self.responsibilities: torch.Tensor
        self.add_state(
            "responsibilities",
            torch.zeros(num_components),
            dist_reduce_fx="sum",
        )

    def update(self, responsibilities: torch.Tensor) -> None:
        # Responsibilities have shape [N, K]
        self.responsibilities.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        return self.responsibilities / self.responsibilities.sum()


class MeanAggregator(Metric):
    """
    The mean aggregator aggregates component means over batches and processes.
    """

    full_state_update = False

    def __init__(
        self,
        num_components: int,
        num_features: int,
    ):
        super().__init__()

        self.mean_sum: torch.Tensor
        self.add_state(
            "mean_sum",
            torch.zeros(num_components, num_features),
            dist_reduce_fx="sum",
        )

        self.component_weights: torch.Tensor
        self.add_state(
            "component_weights",
            torch.zeros(num_components),
            dist_reduce_fx="sum",
        )

    def update(
        self, data: torch.Tensor, responsibilities: torch.Tensor
    ) -> None:
        # Data has shape [N, D]
        # Responsibilities have shape [N, K]
        self.mean_sum.add_(responsibilities.t().matmul(data))
        self.component_weights.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        return self.mean_sum / self.component_weights.unsqueeze(1)


class CovarianceAggregator(Metric):
    """
    The covariance aggregator aggregates component covariances over batches and
        processes.
    """

    full_state_update = False

    def __init__(
        self,
        num_components: int,
        num_features: int,
        reg: float,
    ):
        super().__init__()

        self.num_components = num_components
        self.num_features = num_features
        self.reg = reg

        self.covariance_sum: torch.Tensor

        # Covariance shape
        covariance_shape = torch.Size(
            [self.num_components, self.num_features, self.num_features]
        )

        self.add_state(
            "covariance_sum",
            torch.zeros(covariance_shape),
            dist_reduce_fx="sum",
        )

        self.component_weights: torch.Tensor
        self.add_state(
            "component_weights",
            torch.zeros(num_components),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        data: torch.Tensor,
        responsibilities: torch.Tensor,
        means: torch.Tensor,
    ) -> None:
        data_component_weights = responsibilities.sum(0)
        self.component_weights.add_(data_component_weights)

        # We iterate over each component since this is typically faster...
        for i in range(self.num_components):
            component_diff = data - means[i]
            covars = (
                responsibilities[:, i].unsqueeze(1) * component_diff
            ).T.matmul(component_diff)

            # Add regularization to the diagonal of the covariance matrix
            regularization = self.reg * torch.eye(
                self.num_features, device=covars.device
            )
            covars_regularized = covars + regularization

            self.covariance_sum[i].add_(covars_regularized)

    def compute(self) -> torch.Tensor:
        # covariance_type == "full"
        result = self.covariance_sum / self.component_weights.unsqueeze(
            -1
        ).unsqueeze(-1).add_(1e-16)

        diag_mask = (
            torch.eye(
                self.num_features, device=result.device, dtype=result.dtype
            )
            .bool()
            .unsqueeze(0)
            .expand(self.num_components, -1, -1)
        )
        result[diag_mask] += self.reg
        return result
