# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""New England data module.

Thin subclasses of the LCL data module: the packed data.csv rows
carry the integer conditioning labels produced by preprocess_ne
(month and dayofweek come from the packing step itself), exposed as
long tensors in the fixed config.FEATURE_COLS order.
"""

from pathlib import Path
from typing import Optional

import torch

from opensynth.data_modules.lcl_data_module import LCLData, LCLDataModule
from opensynth.datasets.new_england import config


class NEData(LCLData):
    """
    New England dataset of packed daily load profiles.

    Returns TrainingData with kwh standardised against the training
    statistics and one integer tensor per conditioning feature, in
    the fixed order given by feature_cols.
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        n_samples: int,
        feature_cols: Optional[list[str]] = None,
        outlier_path: Optional[Path] = None,
    ):
        super().__init__(
            data_path=data_path,
            stats_path=stats_path,
            n_samples=n_samples,
            outlier_path=outlier_path,
            feature_cols=(
                list(feature_cols)
                if feature_cols
                else list(config.FEATURE_COLS)
            ),
        )
        # Integer labels as long tensors (LCLData keeps raw pandas
        # series; batching collates both identically, but downstream
        # NE code indexes items directly)
        self.features = {
            col: torch.from_numpy(self.df[col].values).long()
            for col in self.feature_cols
        }


class NEDataModule(LCLDataModule):
    """
    New England data module for Faraday training.
    """

    dataset_cls = NEData
