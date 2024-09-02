# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
from enum import StrEnum
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
import wget


def check_redownload(file_path: Path) -> bool:
    """
    Check if file already exists.
    If it does, ask user if they want to download it again.

    Args:
        file_path (str): file path

    Returns:
        bool: True if user expresses intent
        to download it again, False otherwise
    """
    if path.exists(file_path):
        prompt = f"{file_path} already exists. Redownload? (y/n): "
        return True if input(prompt) == "y" else False
    return True


def download_data(url: str, filename: Path, unzip: bool = False):
    """
    Download data from a URL and save it to a file.
    Function will check if file already exists,
    and if does, prompt user to confirm redownload.

    Args:
        url (str): URL to download the data from
        filename (str): File path to save the downloaded data
    """
    if check_redownload(filename):
        if filename.exists():
            os.remove(filename)
        logging.info(f"Downloading data from: {url}")
        wget.download(url, filename)
    else:
        logging.info("Skipping download")

    if unzip:
        # Note: LCL dataset needs to be unzipped in command line with unzip
        # It's compressed using algorithms that Python's
        # zipfile module can't handle
        subprocess.run(["unzip", filename, "-d", "data/raw"])


def gaussian_noise_generator(
    mean: float, scale: float, mean_factor: float, size: tuple[int, int]
) -> np.array:
    """
    Gaussian noise generator

    Args:
        mean (float): Mean
        scale (float): Standard Deviation
        mean_factor (float): Scaling factor to scale mean by
        size (tuple[int, int]): Outlier dimension

    Returns:
        np.array: Generated outliers
    """
    noise_array = np.random.normal(
        loc=mean * mean_factor, scale=scale, size=size
    )
    return noise_array


def gamma_noise_generator(
    mean: float, scale: float, mean_factor: float, size: tuple[int, int]
) -> np.array:
    """
    Gamma noise generator

    Args:
        mean (float): Mean
        scale (float): Standard Deviation
        mean_factor (float): Scaling factor to scale mean by
        size (tuple[int, int]): Outlier dimension

    Returns:
        np.array: Generated outliers
    """
    noise_array = np.random.gamma(
        shape=mean * mean_factor, scale=scale, size=size
    )
    return noise_array


class NoiseType(StrEnum):
    GAUSSIAN = "gaussian"
    GAMMA = "gamma"


class NoiseFactory:
    def __init__(self, noise_type: NoiseType, **kwargs):
        """
        Factory class to generate noise based on the noise type

        Args:
            noise_type (NoiseType): Noise type.
        """
        self.noise_type = noise_type
        self.kwargs = kwargs

    def generate_noise(self) -> np.array:
        """
        Returns noise based on noise type

        Raises:
            ValueError: If noise type is not supported

        Returns:
            np.array:: Numpy array of random noise
        """
        if self.noise_type not in NoiseType._value2member_map_:
            raise ValueError(f"{self.noise_type} not supported!")
        elif self.noise_type == NoiseType.GAUSSIAN:
            return gaussian_noise_generator(**self.kwargs)
        elif self.noise_type == NoiseType.GAMMA:
            return gamma_noise_generator(**self.kwargs)

    def inject_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject noise into a dataframe. First samples n_outliers
        of rows randomly from dataframe, and overwrite kWh values
        with random noise.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with noisy outliers.
        """
        n_outliers = self.kwargs["size"][0]
        df_out = df.sample(n_outliers)
        noise_array = self.generate_noise()
        df_out["kwh"] = noise_array.tolist()
        df_out["segment"] = str(self.noise_type)
        return df_out
