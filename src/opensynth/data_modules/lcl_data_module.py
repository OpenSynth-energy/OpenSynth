# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 0
g = torch.Generator()
g.manual_seed(RANDOM_STATE)


class LCLData(Dataset):
    """
    Low CarbonLondon Dataset
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        n_samples: int,
        outlier_path: Optional[Path] = None,
    ):
        """
        Args:
            data_path (Path): Data path
            stats_path (Path): Stats path. Note when loading evaluation
            dataset, the stats path point to the stats of training data,
            rather than evaluation data to avoid data leakage!
            n_samples (int): Number of samples to load
            outlier_path (Path, optional): Path to outlier data.
            Defaults to None.
        """
        self.df = pd.read_csv(data_path)
        self.df_stats = pd.read_csv(stats_path)
        self.outlier = True if outlier_path else False

        # Parse stats
        self.feature_mean = self.df_stats["mean"].values[0]
        self.feature_std = self.df_stats["stdev"].values[0]

        # Resample Dataset
        self.n_samples = n_samples
        self.df = self.df.sample(
            self.n_samples, random_state=RANDOM_STATE
        ).reset_index(drop=True)

        # Combine with outliers:
        if self.outlier:
            self.df_outliers = pd.read_csv(outlier_path)
            self.df = pd.concat([self.df, self.df_outliers])
            self.df = self.df.sample(
                frac=1, random_state=RANDOM_STATE
            ).reset_index(drop=True)

        # Parse columns
        self.kwh = self.df["kwh"].apply(ast.literal_eval)
        self.kwh = torch.from_numpy(np.array(self.kwh.tolist())).float()
        self.month = self.df["month"]
        self.dayofweek = self.df["dayofweek"]

    def standardise(self, x: torch.tensor) -> torch.tensor:
        """
        Standardise kWh with mean 0 and std 1

        Args:
            x (torch.tensor): Input kWh

        Returns:
            torch.tensor: Standardised kWh
        """
        return (x - self.feature_mean) / self.feature_std

    def reconstruct(self, xhat: torch.tensor) -> torch.tensor:
        """
        Reconstruct kWh from standardised values

        Args:
            xhat (torch.tensor): standardised kWh

        Returns:
            torch.tensor: reconstructed kWh
        """
        return (xhat * self.feature_std) + self.feature_mean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        standardised_kwh = self.standardise(self.kwh[idx])
        mth = self.month[idx]
        dow = self.dayofweek[idx]

        return standardised_kwh, mth, dow


class LCLDataModule(pl.LightningDataModule):
    """
    Low Carbon London data module
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        batch_size: int,
        n_samples: int,
        outlier_path: Optional[Path] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.outlier_path = outlier_path
        self.outlier = True if outlier_path else False

    def prepare_data(self):
        pass

    def setup(self, stage=""):

        self.dataset = LCLData(
            data_path=self.data_path,
            stats_path=self.stats_path,
            n_samples=self.n_samples,
            outlier_path=self.outlier_path,
        )

        if self.outlier:
            self.outlier_dataset = LCLData(
                data_path=self.outlier_path,
                stats_path=self.stats_path,
                n_samples=100,  # Outlier size = 100
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            self.batch_size,
            drop_last=True,
            shuffle=False,
            generator=g,
        )

    def outlier_dataloader(self):
        return DataLoader(
            self.outlier_dataset,
            100,
            drop_last=True,
            shuffle=False,
            generator=g,
        )

    def reconstruct_kwh(self, xhat: torch.tensor) -> torch.tensor:
        return self.dataset.reconstruct(xhat)
