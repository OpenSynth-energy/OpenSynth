import torch
from torchmetrics import Metric

# class Weights(Metric):


class PriorAggregator(Metric):
    """
    The prior aggregator aggregates component probabilities over batches and
    process.
    """

    full_state_update = False  # Why false

    def __init__(
        self,
        num_components: int,
    ):
        super().__init__()

        self.responsibilities: torch.Tensor
        self.add_state(
            "responsibilities",
            torch.zeros(num_components),
            # Sum across all GPUs to gather
            # Count across all GPUs
            dist_reduce_fx="sum",
        )

    def update(self, responsibilities: torch.Tensor) -> None:
        # Responsibilities have shape [N, K]
        # sum dim=0 gives the total count of samples in each cluster
        self.responsibilities.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        # Calculates the weights of each cluster
        # I.e. proportion of samples in each cluster
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

        # Add all centroid values across all GPUs
        self.mean_sum.add_(responsibilities.t().matmul(data))
        # Add all the count of samples across all GPUs
        self.component_weights.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        # Sum of mean / count of samples gives the average mean
        # across all GPUs
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
        # SUm of covariance matrix across all GPUs
        self.add_state(
            "covariance_sum",
            torch.zeros(covariance_shape),
            dist_reduce_fx="sum",
        )
        # Count of samples in each clusters across all GPUs
        self.component_weights: torch.Tensor
        self.add_state(
            "component_weights",
            torch.zeros(num_components),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        X: torch.Tensor,
        responsibilities: torch.Tensor,
        means: torch.Tensor,
    ) -> None:

        # Count of samples in each cluster. This gives
        # All counts of samples across all GPUs
        data_component_weights = responsibilities.sum(0)
        self.component_weights.add_(data_component_weights)

        # We iterate over each component since this is typically faster...
        for k in range(self.num_components):
            diff = X - means[k]
            covars_k = (
                torch.matmul(responsibilities[:, k] * diff.T, diff)
                / self.component_weights[k]
            )
            covars_k += self.reg
            self.covariance_sum[k].add_(covars_k)

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
