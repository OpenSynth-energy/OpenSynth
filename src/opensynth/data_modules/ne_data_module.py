# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""New England data module.

Structural copy of the LCL data module with a configurable feature
column list. The packed data.csv rows carry the integer conditioning
labels produced by preprocess_ne; month and dayofweek come from the
packing step itself.
"""

import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from opensynth.data_modules.lcl_data_module import (
    RANDOM_STATE,
    TrainingData,
    seed_worker,
)
from opensynth.datasets.new_england import config

g = torch.Generator()
g.manual_seed(RANDOM_STATE)


class NEData(Dataset):
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
        """
        Args:
            data_path (Path): Packed data.csv path.
            stats_path (Path): mean_std.csv of the training split
                (also when loading holdout data, to avoid leakage).
            n_samples (int): Number of daily profiles to sample.
            feature_cols (list[str], optional): Conditioning feature
                order. Defaults to config.FEATURE_COLS.
            outlier_path (Path, optional): outliers.csv to append.
        """
        self.feature_cols = (
            list(feature_cols) if feature_cols else list(config.FEATURE_COLS)
        )
        self.df = pd.read_csv(data_path)
        df_stats = pd.read_csv(stats_path)
        self.feature_mean = df_stats["mean"].values[0]
        self.feature_std = df_stats["stdev"].values[0]

        self.n_samples = n_samples
        self.df = self.df.sample(
            self.n_samples, random_state=RANDOM_STATE
        ).reset_index(drop=True)

        if outlier_path:
            df_outliers = pd.read_csv(outlier_path)
            self.df = pd.concat([self.df, df_outliers])
            self.df = self.df.sample(
                frac=1, random_state=RANDOM_STATE
            ).reset_index(drop=True)

        kwh = self.df["kwh"].apply(ast.literal_eval)
        self.kwh = torch.from_numpy(np.array(kwh.tolist())).float()
        self.features = {
            col: torch.from_numpy(self.df[col].values).long()
            for col in self.feature_cols
        }

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def reconstruct(self, xhat: torch.Tensor) -> torch.Tensor:
        return (xhat * self.feature_std) + self.feature_mean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return TrainingData(
            kwh=self.standardise(self.kwh[idx]),
            features={
                col: self.features[col][idx] for col in self.feature_cols
            },
        )


class NEDataModule(pl.LightningDataModule):
    """
    New England data module for Faraday training.
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        batch_size: int,
        n_samples: int,
        feature_cols: Optional[list[str]] = None,
        outlier_path: Optional[Path] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.feature_cols = feature_cols
        self.outlier_path = outlier_path

    def prepare_data(self):
        pass

    def setup(self, stage=""):
        self.dataset = NEData(
            data_path=self.data_path,
            stats_path=self.stats_path,
            n_samples=self.n_samples,
            feature_cols=self.feature_cols,
            outlier_path=self.outlier_path,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            self.batch_size,
            drop_last=True,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )

    def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
        return self.dataset.reconstruct(xhat)
