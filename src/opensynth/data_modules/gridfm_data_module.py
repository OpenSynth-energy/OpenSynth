# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import ast
import random
import time
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

RANDOM_STATE = 0
g = torch.Generator()
g.manual_seed(RANDOM_STATE)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainingData(TypedDict):
    kwh: torch.Tensor
    features: dict[str, torch.Tensor]


class GridFMData(Dataset):
    """
    GridFM Dataset. The dataset should return
    TrainingData(TypedDict) which contains:
    - kwh: MW data
    - features: Dictionary of features including the temperature

    To use Faraday on custom datasets, your data module
    should also return data in the same format.
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        n_samples: int,
        outlier_path: Optional[Path] = None,
        time_window: Optional[list[str]] = None,
    ):
        """
        Args:
            data_path (Path): Data path
            stats_path (Path): Stats path
            n_samples (int): Number of samples to load
            outlier_path (Path, optional): Path to outlier data, default to None
            time_window (list[str], optional): Dates of the start and end of each time windows 
                if we want to constrain the season/year (mostly for evaluating the conditioning on temperature)
                all time windows will be concatenated
                format of each element of the list : start_date/end_date DD-MM-YYYY/DD-MM-YYYY
                default to None
        """
        self.df = pd.read_csv(data_path)
        self.df_stats = pd.read_csv(stats_path)
        self.outlier = True if outlier_path else False

        # Parse stats
        self.feature_mean = self.df_stats["mean"].values[0]
        self.feature_std = self.df_stats["stdev"].values[0]
        self.temperature_mean = self.df_stats["temperature_mean"].values[0]
        self.temperature_std = self.df_stats["temperature_std"].values[0]
        

        # Resample Dataset
        self.n_samples = n_samples
        self.time_window = time_window
        if self.time_window is not None:
            df_ = pd.DataFrame()
            for window in self.time_window:
                start_window = pd.Timestamp(window[:10]).replace(hour=00, minute=00, second=00)
                end_window = pd.Timestamp(window[-10:]).replace(hour=23, minute=59, second=59)
                df_ = pd.concat([df_, 
                                 self.df.loc[(pd.to_datetime(self.df['date']) >= start_window) 
                                    & (pd.to_datetime(self.df['date']) <= end_window)]])
            self.df = df_
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
        raw_temperature = self.df["temperature"].apply(ast.literal_eval)
        raw_temperature = torch.from_numpy(np.array(raw_temperature.tolist())).float()
        self.temperature = self.standardise_temperature(raw_temperature)
        self.month = self.df["month"].values
        day = self.df["dayofweek"].values.astype(int)  # 0-6
        one_hot = np.eye(7)[day]  # shape: (n_samples, 7)
        self.dayofweek = torch.from_numpy(one_hot).float()

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardise MW with mean 0 and std 1

        Args:
            x (torch.Tensor): Input MW

        Returns:
            torch.Tensor: Standardised MW
        """
        return (x - self.feature_mean) / self.feature_std

    def reconstruct(self, xhat: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct MW from standardised values

        Args:
            xhat (torch.Tensor): standardised MW

        Returns:
            torch.Tensor: reconstructed MW
        """
        return (xhat * self.feature_std) + self.feature_mean

    def standardise_temperature(self, temp: torch.Tensor) -> torch.Tensor:
        """
        Standardise temperature with mean 0 and std 1

        Args:
            temp (torch.Tensor): Input temperature

        Returns:
            torch.Tensor: Standardised temperature
        """
        return (temp - self.temperature_mean) / self.temperature_std
    
    def reconstruct_temperature(self, temp_stand: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct temperature from standardised values

        Args:
            temp_stand (torch.Tensor): Standardised temperature

        Returns:
            torch.Tensor: Reconstructed temperature
        """
        return (temp_stand * self.temperature_std) + self.temperature_mean
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        standardised_kwh = self.standardise(self.kwh[idx])
        features: dict[str, torch.Tensor] = {
            "temperature": self.temperature[idx],
            "dayofweek": self.dayofweek[idx]
        }
        return TrainingData(kwh=standardised_kwh, features=features)


class GridFMDataModule(pl.LightningDataModule):
    """
    GridFM data module
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        batch_size: int,
        n_samples: int,
        outlier_path: Optional[Path] = None,
        time_window: Optional[str] = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.outlier_path = outlier_path
        self.outlier = True if outlier_path else False
        self.time_window = time_window

    def prepare_data(self):
        pass

    def setup(self, stage=""):

        self.dataset = GridFMData(
            data_path=self.data_path,
            stats_path=self.stats_path,
            n_samples=self.n_samples,
            outlier_path=self.outlier_path,
            time_window=self.time_window
        )

        if self.outlier:
            self.outlier_dataset = GridFMData(
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
            worker_init_fn=seed_worker,
        )

    def outlier_dataloader(self):
        return DataLoader(
            self.outlier_dataset,
            100,
            drop_last=True,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )

    def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
        return self.dataset.reconstruct(xhat)
    
    def standardise_temperature(self, temp: torch.Tensor) -> torch.Tensor:
        return self.dataset.standardise_temperature(temp)
    
    def reconstruct_temperature(self, temp_stand: torch.Tensor) -> torch.Tensor:
        return self.dataset.reconstruct_temperature(temp_stand)