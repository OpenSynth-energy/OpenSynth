# SPDX-License-Identifier: MIT
import ast
from pathlib import Path

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
        feature_mean: float,
        feature_std: float,
        n_samples: int,
    ):
        self.df = pd.read_csv(data_path)
        self.n_samples = n_samples
        self.df = self.df.sample(
            self.n_samples, random_state=RANDOM_STATE
        ).reset_index(drop=True)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
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
        batch_size: int,
        feature_mean: float,
        feature_std: float,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def prepare_data(self):
        pass

    def setup(self, stage=""):
        self.dataset = LCLData(
            data_path=self.data_path,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            self.batch_size,
            drop_last=True,
            shuffle=False,
            generator=g,
        )

    def reconstruct_kwh(self, xhat: torch.tensor) -> torch.tensor:
        return self.dataset.reconstruct(xhat)
