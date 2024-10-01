# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""
Code based on source: Borchert, O. (2022). PyCave (Version 3.2.1)
[Computer software] https://pycave.borchero.com/
"""

import random

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
        covariance_type: str,
        reg: float,
    ):
        super().__init__()

        self.num_components = num_components
        self.num_features = num_features
        self.covariance_type = covariance_type
        self.reg = reg

        self.covariance_sum: torch.Tensor

        # Covariance shape
        if covariance_type == "full":
            covariance_shape = torch.Size(
                [self.num_components, self.num_features, self.num_features]
            )
        if covariance_type == "tied":
            covariance_shape = torch.Size(
                [self.num_features, self.num_features]
            )
        if covariance_type == "diag":
            covariance_shape = torch.Size(
                [self.num_components, self.num_features]
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

        if self.covariance_type in ("spherical", "diag"):
            x_prob = torch.matmul(responsibilities.t(), data.square())
            m_prob = data_component_weights.unsqueeze(-1) * means.square()
            xm_prob = means * torch.matmul(responsibilities.t(), data)
            covars = x_prob - 2 * xm_prob + m_prob
            if self.covariance_type == "diag":
                self.covariance_sum.add_(covars)
            else:  # covariance_type == "spherical"
                self.covariance_sum.add_(covars.mean(1))
        elif self.covariance_type == "tied":
            # This is taken from
            # https://github.com/scikit-learn/scikit-learn/blob/
            # 844b4be24d20fc42cc13b957374c718956a0db39/sklearn/mixture/
            # _gaussian_mixture.py#L183
            x_sq = data.T.matmul(data)
            mean_sq = (data_component_weights * means.T).matmul(means)
            self.covariance_sum.add_(x_sq - mean_sq)
        else:  # covariance_type == "full":
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
        if self.covariance_type == "diag":
            return (
                self.covariance_sum / self.component_weights.unsqueeze(-1)
                + self.reg
            )
        if self.covariance_type == "spherical":
            return (
                self.covariance_sum / self.component_weights
                + self.reg * self.num_features
            )
        if self.covariance_type == "tied":
            result = self.covariance_sum / self.component_weights.sum()
            shape = result.size()
            result = result.flatten()
            result[:: self.num_features + 1].add_(self.reg)
            return result.view(shape)
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


# -----------------------------------------------------------------------------
# K-Means Metrics
# -----------------------------------------------------------------------------
class CentroidAggregator(Metric):
    """
    The centroid aggregator aggregates kmeans centroids over batches and
        processes.
    """

    full_state_update = False

    def __init__(
        self,
        num_clusters: int,
        num_features: int,
    ):
        super().__init__()

        self.num_clusters = num_clusters
        self.num_features = num_features

        self.centroids: torch.Tensor
        self.add_state(
            "centroids",
            torch.zeros(num_clusters, num_features),
            dist_reduce_fx="sum",
        )

        self.cluster_counts: torch.Tensor
        self.add_state(
            "cluster_counts", torch.zeros(num_clusters), dist_reduce_fx="sum"
        )

    def update(self, data: torch.Tensor, assignments: torch.Tensor) -> None:
        indices = assignments.unsqueeze(1).expand(-1, self.num_features)

        self.centroids.scatter_add_(0, indices, data)

        counts = assignments.bincount(minlength=int(self.num_clusters)).float()
        self.cluster_counts.add_(counts)

    def compute(self) -> torch.Tensor:
        return self.centroids / self.cluster_counts.unsqueeze(-1)


class UniformSampler(Metric):
    """
    The uniform sampler randomly samples a specified number of datapoints
        uniformly from all datapoints.

    The idea is the following: sample the number of choices from each batch and
        track the number of datapoints that was already sampled from. When
        sampling from the union of existing choices and a new batch, more
        weight is put on the existing choices (according to the number of
        datapoints they were already sampled from).
    """

    full_state_update = False

    def __init__(
        self,
        num_choices: int,
        num_features: int,
    ):
        super().__init__()

        self.num_choices = num_choices

        self.choices: torch.Tensor
        self.add_state(
            "choices",
            torch.empty(num_choices, num_features),
            dist_reduce_fx="cat",
        )

        self.choice_weights: torch.Tensor
        self.add_state(
            "choice_weights", torch.zeros(num_choices), dist_reduce_fx="cat"
        )

    def update(self, data: torch.Tensor) -> None:
        if self.num_choices == 1:
            # If there is only one choice, the fastest thing is to use the
            # `random` package. The cumulative weight of the data is its size,
            # the cumulative weight of the current choice is some value.
            cum_weight = data.size(0) + self.choice_weights.item()
            if random.random() * cum_weight < data.size(0):
                # Use some item from the data, else keep the current choice
                self.choices.copy_(data[random.randrange(data.size(0))])
        else:
            # The choices are computed from scratch every time, weighting the
            # current choices by the cumulative weight put on them
            weights = torch.cat(
                [
                    torch.ones(
                        data.size(0), device=data.device, dtype=data.dtype
                    ),
                    self.choice_weights,
                ]
            )
            pool = torch.cat([data, self.choices])
            samples = weights.multinomial(self.num_choices)
            self.choices.copy_(pool[samples])

        # The weights are the cumulative counts over the number of choices
        self.choice_weights.add_(data.size(0) / self.num_choices)

    def compute(self) -> torch.Tensor:
        # In the ddp setting, there are "too many" choices, so we sample
        if self.choices.size(0) > self.num_choices:
            samples = self.choice_weights.multinomial(self.num_choices)
            return self.choices[samples]
        return self.choices
