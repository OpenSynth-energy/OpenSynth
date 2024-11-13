# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from typing import Union

import litdata as ld
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from opensynth.data_modules.lcl_data_module import TrainingData

RANDOM_STATE = 0
g = torch.Generator()
g.manual_seed(RANDOM_STATE)


class StreamData(ld.StreamingDataset):
    """
    Streaming Dataset. The dataset should return
    TrainingData(TypedDict) which contains:
    - kwh: kWh data
    - features: Dictionary of features

    To use Faraday on custom datasets, your data module
    should also return data in the same format.

    For more information on the attributes used for the StreamingDataset, see:
    https://github.com/Lightning-AI/litdata/blob/main/src/litdata/streaming/dataset.py#L43
    """

    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        storage_options: Union[dict, None] = None,
        subsample: float = 1.0,
        shuffle: bool = False,
        drop_last: bool = True,
        max_pre_download: int = 2,
        max_cache_size: str = "100GB",
    ):
        """
        Args:
            data_path (str): Path to stream data from (can be GCS URI or local
                path)
            stats_path (Path): Stats path. Note when loading evaluation
                dataset, the stats path point to the stats of training data,
                rather than evaluation data to avoid data leakage!
            storage_options (dict): Storage options metadata (e.g. GCS project)
            feature_mean (float): Mean of consumption data (kWh), used for
                standardisation
            feature_std (float): STD of consumption data (kWh), used for
                standardisation
            shuffle (bool, optional): Whether to shuffle the data
            drop_last (bool, optional): Drop last items so processes return
                same amount of data
        """
        super().__init__(
            input_dir=data_path,
            storage_options=storage_options,
            shuffle=shuffle,
            drop_last=drop_last,
            subsample=subsample,
            max_pre_download=max_pre_download,
            max_cache_size=max_cache_size,
        )
        # Parse stats
        self.df_stats = pd.read_csv(stats_path)
        self.feature_mean = self.df_stats["mean"].values[0]
        self.feature_std = self.df_stats["stdev"].values[0]

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardise kWh with mean 0 and std 1
        Args:
            x (torch.Tensor): Input kWh
        Returns:
            torch.Tensor: Standardised kWh
        """
        return (x - self.feature_mean) / self.feature_std

    def reconstruct(self, xhat: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct kWh from standardised values
        Args:
            xhat (torch.Tensor): standardised kWh
        Returns:
            torch.Tensor: reconstructed kWh
        """
        return (xhat * self.feature_std) + self.feature_mean

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Parse kwh
        kwh = ast.literal_eval(data["kwh"])
        kwh = torch.from_numpy(np.array(kwh)).float()
        standardised_kwh = self.standardise(kwh)

        return TrainingData(
            kwh=standardised_kwh,
            features=data["features"],
        )


class StreamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        stats_path: Path,
        storage_options: Union[dict, None] = None,
        num_workers: int = 0,
        batch_size: int = 5000,
        subsample: float = 1.0,
        max_cache_size: str = "100GB",
        profile_batches: Union[int, None] = None,
        shuffle: bool = False,
        drop_last: bool = True,
        persistent_workers: bool = False,
        pin_memory: bool = True,
    ):
        """
        Faraday Streaming Lightning Data Module.

        For more information on the attributes used for the
        StreamingDataLoader, see:
        https://github.com/Lightning-AI/litdata/blob/main/src/litdata/streaming/dataloader.py#L494

        Args:
            data_path (str): Input streaming directory (GCS or local)
            stats_path (Path): Stats path. Note when loading evaluation
                dataset, the stats path point to the stats of training data,
                rather than evaluation data to avoid data leakage!
            storage_options (dict): Storage options metadata (e.g. GCS project)
            num_workers (int, optional): How many subprocesses to use for
                dataloading
            batch_size (int, optional): Samples per batch. Defaults to 5000
            subsample (float, optional): Float representing fraction of the
                dataset to be randomly sampled
            profile_batches (int, optional): Whether to record data loading
                profile and generate a result.json file (for debugging).
                Value is the number of batches to profile. Defaults to None.
            max_cache_size (str, optional): The maximum cache size used by
                the StreamingDataset
            shuffle (bool, optional): Whether to shuffle the data
            drop_last (bool, optional): Drop last items so processes return
                same amount of data
            persistent_workers (bool, optional): Whether to keep workers
                alive between epochs. Defaults to False
            pin_memory (bool, optional): Whether to pin memory in DataLoader.
                Defaults to True
        """
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.storage_options = storage_options
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.subsample = subsample
        self.profile_batches = profile_batches
        self.max_cache_size = max_cache_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        pass

    def setup(self, stage: str = ""):
        self.train_dataset = StreamData(
            data_path=self.data_path,
            stats_path=self.stats_path,
            storage_options=self.storage_options,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            subsample=self.subsample,
            max_cache_size=self.max_cache_size,
        )

    def train_dataloader(self):
        return ld.StreamingDataLoader(
            dataset=self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            persistent_workers=self.persistent_workers,
            profile_batches=self.profile_batches,
            pin_memory=self.pin_memory,
        )

    def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
        return self.train_dataset.reconstruct(xhat)
