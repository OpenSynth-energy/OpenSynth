# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric


class SufficientStatisticMetric(Metric):
    """Accumulates a GMM sufficient statistic tensor by summation, both
    across mini-batches within an epoch and, in distributed training,
    across devices.
    """

    full_state_update = False

    def __init__(self, shape: torch.Size):
        super().__init__()
        self.value: torch.Tensor
        self.add_state("value", torch.zeros(shape), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor) -> None:
        self.value += value

    def compute(self) -> torch.Tensor:
        return self.value


class NegativeLogLikelihoodMetric(Metric):
    """Accumulates the mean negative log likelihood across mini-batches
    within an epoch, and across devices in distributed training.
    """

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.nll: torch.Tensor
        self.count: torch.Tensor
        self.add_state("nll", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("count", torch.zeros(1), dist_reduce_fx="sum")

    def update(self, nll: torch.Tensor) -> None:
        self.nll += nll
        self.count += 1

    def compute(self) -> torch.Tensor:
        return self.nll / self.count
