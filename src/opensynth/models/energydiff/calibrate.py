# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
#
# The EnergyDiff model is made available via Nan Lin and Pedro P. Vergara
# from the Delft University of Technology. Nan Lin and Pedro P. Vergara are
# funded via the ALIGN4Energy Project (with project number NWA.1389.20.251) of
# the research programme NWA ORC 2020 which is (partly) financed by the Dutch
# Research Council (NWO), The Netherland.
import enum
from typing import Tuple

import torch
from einops import rearrange


class DataShapeType(enum.Enum):
    """
    Enum class for data shape
    """

    BATCH_CHANNEL_SEQUENCE = "batch_channel_sequence"
    BATCH_SEQUENCE = "batch_sequence"
    CHANNEL_SEQUENCE = "channel_sequence"
    SEQUENCE = "sequence"


def _transform_to_target_shape(
    x: torch.Tensor,
    target_shape: DataShapeType,
    num_channel: int | None = None,
) -> torch.Tensor:
    """transform intermediate data to target shape

    Arguments:
        x -- intermediate data of shape BATCH (CHANNEL*)SEQUENCE
        target_shape -- DataShapeType
    """
    if (
        target_shape
        in [
            DataShapeType.CHANNEL_SEQUENCE,
            DataShapeType.BATCH_CHANNEL_SEQUENCE,
        ]
        and num_channel is None
    ):
        raise ValueError("num_channel should be provided")

    if target_shape == DataShapeType.BATCH_CHANNEL_SEQUENCE:
        x = rearrange(x, "b (c l) -> b c l", c=num_channel)
    elif target_shape == DataShapeType.CHANNEL_SEQUENCE:
        x = rearrange(x, "1 (c l) -> c l", c=num_channel)
    elif target_shape == DataShapeType.SEQUENCE:
        x = rearrange(x, "1 cl -> cl")

    return x


def _transform_to_process_shape(
    y: torch.Tensor, seq_length: int
) -> Tuple[torch.Tensor, DataShapeType, int | None]:
    """transform input data into process shape (BATCH SEQUENCE)

    Arguments:
        y -- torch.Tensor, of shape (batch dim1 dim2) or (batch dim1)
        seq_length -- int, equal to dim1 * dim2 (batch dim1 dim2) \
            or dim1 (batch dim1)

    Returns:
        y -- torch.Tensor of shape (batch dim)
        input_shape_type -- DataShapeType, shape of original input data y
        num_in_channel -- int|None, number of channels in the input data y
    """
    yndim = len(y.shape)
    num_in_channel = None
    if yndim == 3:
        input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
        num_in_channel, _ = y.shape[1], y.shape[2]
        y = rearrange(y, "b c l -> b (c l)")
    elif yndim == 2:
        if y.shape[0] * y.shape[1] == seq_length:
            input_shape_type = DataShapeType.CHANNEL_SEQUENCE
            num_in_channel, _ = y.shape[0], y.shape[1]
            y = rearrange(y, "c l -> 1 (c l)")
        else:
            input_shape_type = DataShapeType.BATCH_SEQUENCE
            pass
    elif yndim == 1:
        input_shape_type = DataShapeType.SEQUENCE
        y = rearrange(y, "cl -> 1 cl")
    else:
        raise ValueError("Invalid input shape")

    return y, input_shape_type, num_in_channel


class MultiDimECDF:
    """
    Batched ECDF estimation.
    """

    def __init__(self, x):
        """
        Args:
            x: input data, shape (batch, channel, sequence) \
                or (batch, sequence)
                NOTE: NOT ALLOWED: (channel, sequence) or (sequence) \
                    because at you need at least a batch to calculate ECDF
        """
        # super().__init__()
        if len(x.shape) == 2:  # (batch, sequence) / (batch, channel*sequence)
            self.dim = x.shape[1]
            x = x
        else:  # (batch, channel, sequence)
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = x
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(
            x, dim=0
        )  # expected: DataShapeType.BATCH_SEQUENCE
        # self.x_sorted = self.stable(self.x_sorted)

    @property
    def cdf(self) -> Tuple[torch.Tensor, torch.Tensor]:
        "returns x, cdf"
        x, counts = torch.unique(
            self.x_sorted.contiguous(), dim=0, sorted=True, return_counts=True
        )  # shape (n_unique, dim)
        events = torch.cumsum(counts, dim=0)  # shape (n_unique, dim)
        cdf = events.float() / self.num_sample  # shape (n_unique, dim)
        return x, cdf

    def to(self, device):
        self.x = self.x.to(device)
        self.x_sorted = self.x_sorted.to(device)
        return self

    def transform(self, x):
        """
        Args:
            x: input data, shape (b, c, l) or (b, c*l) or (c, l) or (c*l)
        (x_sorted shape) (b, c*l)
        Returns:
            y: ECDF transformed data, shape (batch_size, dim1, dim2)
                or (batch_size, dim)
        """
        x_sorted = self.x_sorted.to(x.device)
        xndim = len(x.shape)
        input_shape_type = DataShapeType.BATCH_SEQUENCE
        num_channel = None
        if xndim == 3:
            input_shape_type = DataShapeType.BATCH_CHANNEL_SEQUENCE
            num_channel, _ = x.shape[1], x.shape[2]
            x = rearrange(x, "b c l -> b (c l)")
        elif xndim == 2:
            if x.shape[0] * x.shape[1] == self.x_sorted.shape[1]:
                input_shape_type = DataShapeType.CHANNEL_SEQUENCE
                num_channel, _ = x.shape[0], x.shape[1]
                x = rearrange(x, "c l -> 1 (c l)")
            else:
                input_shape_type = DataShapeType.BATCH_SEQUENCE
                pass
        elif xndim == 1:
            input_shape_type = DataShapeType.SEQUENCE
            x = rearrange(x, "cl -> 1 cl")
        else:
            raise ValueError("Invalid input shape")

        indices = torch.searchsorted(
            x_sorted.transpose(0, 1).contiguous(),
            x.transpose(0, 1).contiguous(),
            side="right",
        ).transpose(
            0, 1
        )  # [b, c*l]
        cdf_values = indices.float() / self.num_sample

        y = torch.clamp(cdf_values, 0.0, 1.0)

        y = _transform_to_target_shape(
            y, input_shape_type, num_channel=num_channel
        )

        return y

    def inverse_transform(self, y):
        """
        Args:
            y: ECDF transformed data,
                shape (batch_size, dim1, dim2) or (batch_size, dim)
        Returns:
            x: input data, shape (batch_size, dim)
        """
        x_sorted = self.x_sorted.to(y.device)
        y, input_shape_type, num_in_channel = _transform_to_process_shape(
            y,
            seq_length=x_sorted.shape[1],
        )  # y shape: (batch_size, dim)

        y = torch.clamp(y, 0.0, 1.0)
        y_scaled = y * self.num_sample
        indices_lower = torch.floor(y_scaled).long()  # range (0, num_sample)
        xlower = torch.gather(
            x_sorted, 0, indices_lower.clamp(min=1, max=self.num_sample) - 1
        )  # range bounded by x_sorted
        x = xlower

        x = _transform_to_target_shape(x, input_shape_type, num_in_channel)

        return x

    def re_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 2:
            self.x = x
            self.dim = x.shape[1]
        else:
            self.x = rearrange(x, "n d1 d2 -> n (d1 d2)")
            self.dims = (x.shape[1], x.shape[2])
            self.dim = x.shape[1] * x.shape[2]
        self.num_sample = x.shape[0]
        self.x_sorted, _ = torch.sort(x, dim=0)

    def continue_estimate(self, x):
        """
        Args:
            x: input data, shape (n_samples, dim)
        """
        self.x = self.x.to(x.device)
        if len(x.shape) == 3:
            x = rearrange(x, "n d1 d2 -> n (d1 d2)")
        self.x = torch.cat([self.x, x], dim=0)
        self.num_sample = self.x.shape[0]
        self.x_sorted, _ = torch.sort(self.x, dim=0)

    def __call__(self, x):
        """get ecdf of x"""
        x = self.transform(x)

        return x


def calibrate(
    target,  # [batch sequence]
    source,  # [batch sequence]
):
    """
    calibrate the marginals of source data to target data.
        return calibrated source data.
    """
    ecdf_source = MultiDimECDF(source)
    ecdf_target = MultiDimECDF(target)
    calibrated_source = ecdf_target.inverse_transform(
        ecdf_source.transform(source)
    )

    return calibrated_source
