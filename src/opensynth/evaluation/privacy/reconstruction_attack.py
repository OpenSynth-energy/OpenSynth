# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.evaluation.privacy import generate_attack_data
from opensynth.models.faraday import FaradayModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReconstructionAttackDataset:
    synthetic_samples: torch.Tensor
    real_samples: torch.Tensor
    outlier_samples: torch.Tensor


def create_attack_dataset(
    model: FaradayModel,
    dm: LCLDataModule,
    n_samples: int = 20000,
) -> ReconstructionAttackDataset:
    """
    Generate reconstruction attack dataset

    Args:
        model (FaradayModel): Synthetic model
        dm (LCLDataModule): LCL data module
        n_samples (int): Number of samples to generate. Defaults to 20K.

    Returns:
        ReconstructionAttackDataset: Reconstruction attack data.
    """
    synthetic_samples = generate_attack_data.generate_synthetic_samples(
        model, dm, n_samples
    )

    real_samples = generate_attack_data.draw_real_data(
        dm, n_samples, outliers=False
    )

    outlier_samples = generate_attack_data.draw_real_data(
        dm, n_samples, outliers=True
    )

    return ReconstructionAttackDataset(
        synthetic_samples=synthetic_samples,
        real_samples=real_samples,
        outlier_samples=outlier_samples,
    )


def _calculate_pairwise_euclidean(
    vec1: torch.tensor, vec2: torch.tensor
) -> torch.tensor:
    """
        Calculates pairwise euclidean distance. Creates a matrix of
        Size(Vec1) X Size(Vec2)
    Args:
        vec1 (torch.tensor): Vector 1
        vec2 (torch.tensor): Vector 2

    Returns:
        torch.tensor: Pairwise euclidean distance matrix
    """
    logger.info("Calculating pairwise euclidean distance..")
    euc_pairwise = torch.cdist(vec1, vec2)
    return euc_pairwise


def _convert_wide_to_long(
    matrix: torch.tensor, metric_name: str
) -> pd.DataFrame:
    """
    Converts euclidean matrix (wide table) to long table of
    pairwise distances

    Args:
        matrix (torch.tensor): Pairwise euclidean distance matrix
        metric_name (str): Metric name

    Returns:
        pd.DataFrame: A long dataframe consisting of the seen outlier
        and its pairwise distances to all generated data.
    """
    logger.info("Converting wide to long table..")
    df = pd.DataFrame(matrix.detach().numpy())
    df.columns = [f"{metric_name}{i+1}" for i in range(matrix.shape[1])]
    df["real_outlier"] = df.index + 1

    df_out = pd.wide_to_long(
        df, stubnames=metric_name, i=["real_outlier"], j="synthetic_output"
    )
    df_out = df_out.reset_index()
    df_out = df_out.sort_values(
        by=["real_outlier", "synthetic_output"], ascending=True
    )
    return df_out


def _calculate_vector_norm(data: torch.tensor) -> pd.DataFrame:
    """
    Calculate norm of outlier vector

    Args:
        data (torch.tensor): Input data

    Returns:
        pd.DataFrame: A long dataframe consisting of the seen outlier
        and its pairwise distances to all generated data.
    """
    logger.info("Calculating vector norm..")
    df_vector_norm = pd.DataFrame(columns=["real_outlier"])
    df_vector_norm["real_outlier"] = [i + 1 for i in range(len(data))]
    df_vector_norm["norm"] = torch.linalg.vector_norm(data, dim=1)
    return df_vector_norm


def calculate_distance_norm(
    real: torch.tensor, fake: torch.tensor, group_min: bool = True
) -> pd.DataFrame:
    """
    Calculates distance norms between generated data and seen outliers.

    Args:
        real (torch.tensor): Seen outlier
        fake (torch.tensor): Generated data
        group_min (bool, optional): Group pairwise distances to
        return only the nearest neighbour. Defaults to True. If
        False, returns all pairwise distances.

    Returns:
        pd.DataFrame: Pandas dataframe of outliers to its
        neighbour. Distance is expressed as a ratio of
        the outlier's vector norm.
    """
    matrix = _calculate_pairwise_euclidean(real, fake)
    df_euc = _convert_wide_to_long(matrix, metric_name="euc_dist")
    df_norm = _calculate_vector_norm(real)

    df_dist = df_euc.merge(df_norm, on="real_outlier", how="left")
    df_dist["ratio"] = df_dist["euc_dist"] / df_dist["norm"]
    df_dist = df_dist.sort_values(by="ratio", ascending=True)

    if group_min:
        df_dist = (
            df_dist.groupby(["real_outlier"])["ratio"].min().reset_index()
        )

    return df_dist


def plot_ecdf(df_ecdf, hue_col: str | None = None):
    """
    Plots the cumulative distritution function for reconstruction
    attack.

    Args:
        df_ecdf (_type_): Dataframe
        hue_col (str, optional): Column to groupby. Defaults to None.
    """
    sns.ecdfplot(data=df_ecdf, x="ratio", hue=hue_col)
    plt.xlabel("Ratio of Euclidean Distance to Vector Norm")
    plt.ylabel("Proportion")
    plt.xlim(0, 1)
