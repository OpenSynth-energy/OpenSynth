# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import ast
import logging
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.datasets.low_carbon_london.preprocess_lcl import create_outliers
from opensynth.evaluation.privacy import generate_attack_data
from opensynth.models.faraday import FaradayModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MembershipInferenceAttackSamples:
    synthetic_samples: torch.Tensor
    train_samples: torch.Tensor
    holdout_samples: torch.Tensor
    outlier_seen_samples: torch.Tensor
    outlier_unseen_same_samples: torch.Tensor
    outlier_unseen_diff_samples: torch.Tensor


def _create_unseen_outliers(df: pd.DataFrame, mean: float) -> torch.tensor:
    """
    Creates unseen outliers. Calls unpon the method
    opensynth.datasets.low_carbon_london.preprocess_lcl.create_outliers.
    Creates outliers based on gamma and gaussian distribution
    around the `mean` value.

    Args:
        df (pd.DataFrame): Input dataframe to create outliers from.
        mean (float): Mean value for gaussian and gamma distribution.

    Returns:
        torch.tensor: Generated unseen outliers.
    """
    outliers = create_outliers(df, mean)
    return torch.from_numpy(np.array(outliers["kwh"].values.tolist()))


def _create_attack_samples(
    model: FaradayModel,
    dm_train: LCLDataModule,
    dm_holdout: LCLDataModule,
    n_samples: int = 20000,
) -> MembershipInferenceAttackSamples:
    """
    Creates attack samples for membership inference attack:

    Synthetic samples: generated from the trained model.
    Train samples: real data from the training set (with
        outliers injected.)
    Holdout samples: real data from the holdout set.
    Outlier seen samples: Outliers seen during training.
    Outlier unseen same samples: Outliers not seen during training,
        but from the same distribution as seen outliers.
    Outlier unseen diff samples: Outliers not seen during training,
        but from a different distribution to the seen outliers.

    Args:
        model (FaradayModel): Model to generate synthetic samples.
        dm_train (LCLDataModule): Train data module
        dm_holdout (LCLDataModule): Holdout data module
        n_samples (int, optional): Number of samples. Defaults to 20000.

    Returns:
        MembershipInferenceAttackSamples: Attack samples
    """
    synthetic_samples = generate_attack_data.generate_synthetic_samples(
        model, dm_train, n_samples=n_samples
    )
    train_samples = generate_attack_data.draw_real_data(
        dm=dm_train, n_samples=n_samples
    )
    holdout_samples = generate_attack_data.draw_real_data(
        dm=dm_holdout, n_samples=n_samples
    )
    outlier_samples = generate_attack_data.draw_real_data(
        dm=dm_train, outliers=True, n_samples=n_samples
    )

    df_train = dm_train.dataset.df[
        dm_train.dataset.df["segment"].isna()
    ].copy()
    outlier_unseen_same_samples = _create_unseen_outliers(
        df_train, dm_train.dataset.feature_mean
    )
    outlier_unseen_diff_samples = _create_unseen_outliers(df_train, mean=0)

    return MembershipInferenceAttackSamples(
        synthetic_samples=synthetic_samples,
        train_samples=train_samples,
        holdout_samples=holdout_samples,
        outlier_seen_samples=outlier_samples,
        outlier_unseen_same_samples=outlier_unseen_same_samples,
        outlier_unseen_diff_samples=outlier_unseen_diff_samples,
    )


def _create_mia_train_dataset(
    samples: MembershipInferenceAttackSamples,
) -> pd.DataFrame:
    """
    Create training dataset for membership inference attack.
    This dataset is used to train the MIA discriminator

    Args:
        samples (MembershipInferenceAttackSamples): MIA samples.

    Returns:
        pd.DataFrame: Training dataset for MIA.
    """
    df = pd.DataFrame()

    holdout_data = samples.holdout_samples.detach().numpy().tolist()
    synth_data = samples.synthetic_samples.detach().numpy().tolist()

    holdout_target = [0] * len(holdout_data)
    synth_target = [1] * len(synth_data)

    df["tensors"] = holdout_data + synth_data
    df["target"] = holdout_target + synth_target
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def _create_mia_attack_dataset(
    samples: MembershipInferenceAttackSamples,
) -> pd.DataFrame:
    """
    Create attack dataset for membership inference attack.
    This dataset is used to launch the MIA on the seen outliers.


    Args:
        samples (MembershipInferenceAttackSamples): MIA samples.

    Returns:
        pd.DataFrame: Attack dataset for MIA
    """
    df = pd.DataFrame()

    seen_data = samples.outlier_seen_samples.detach().numpy().tolist()
    unseen_same_data = (
        samples.outlier_unseen_same_samples.detach().numpy().tolist()
    )
    unseen_diff_data = (
        samples.outlier_unseen_diff_samples.detach().numpy().tolist()
    )

    seen_target = [1] * len(seen_data)
    unseen_same_target = [0] * len(unseen_same_data)
    unseen_diff_target = [0] * len(unseen_diff_data)

    seen_type = ["seen"] * len(seen_data)
    unseen_same_type = ["unseen_same"] * len(unseen_same_data)
    unseen_diff_type = ["unseen_diff"] * len(unseen_diff_data)

    df["tensors"] = seen_data + unseen_same_data + unseen_diff_data
    df["target"] = seen_target + unseen_same_target + unseen_diff_target
    df["type"] = seen_type + unseen_same_type + unseen_diff_type
    df = df.sample(frac=1).reset_index(drop=True)

    return df


class MembershipInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        Membership Inference Dataset

        Args:
            df (pd.DataFrame): Input data
        """
        self.df = df
        self.vector = self.df["tensors"].astype(str).apply(ast.literal_eval)
        self.vector = torch.from_numpy(np.array(self.vector.tolist())).float()
        self.label = self.df["target"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vector = self.vector[idx]
        label = self.label[idx]
        return vector, label


class MembershipInferenceDataModule(pl.LightningDataModule):

    def __init__(
        self,
        model: FaradayModel,
        dm_train: LCLDataModule,
        dm_holdout: LCLDataModule,
        batch_size: int,
    ):
        """
        Membership Inference Attack Data Module

        Args:
            model (FaradayModel): Model to generate synthetic samples.
            dm_train (LCLDataModule): Train data module with outliers injected
            dm_holdout (LCLDataModule): Holdout data module
            batch_size (int): Batch size for MIA.
        """

        super().__init__()
        self.model = model
        self.dm_train = dm_train
        self.df_holdout = dm_holdout
        self.samples = _create_attack_samples(
            model=model, dm_train=dm_train, dm_holdout=dm_holdout
        )
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=""):

        # Create training and attack datasets
        df_train = _create_mia_train_dataset(self.samples)
        df_attack = _create_mia_attack_dataset(self.samples)

        # Split df_train into train/ eval to prevent overfitting
        df_train_train, df_train_eval = train_test_split(
            df_train, train_size=0.8, random_state=100
        )

        # Reshuffling data
        df_train_train = df_train_train.sample(frac=1, random_state=100)
        df_train_eval = df_train_eval.sample(frac=1, random_state=100)
        df_attack = df_attack.sample(frac=1, random_state=100)

        self.df_train = df_train_train.reset_index(drop=True)
        self.df_eval = df_train_eval.reset_index(drop=True)
        self.df_attack = df_attack.reset_index(drop=True)

        # Create datasets which will be used to instantiate data loaders
        self.train_dataset = MembershipInferenceDataset(df=self.df_train)
        self.eval_dataset = MembershipInferenceDataset(df=self.df_eval)
        self.attack_dataset = MembershipInferenceDataset(df=self.df_attack)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, self.batch_size, drop_last=True, shuffle=True
        )

    def eval_dataloader(self):
        return DataLoader(
            self.eval_dataset, self.batch_size, drop_last=True, shuffle=False
        )

    def attack_dataloader(self):
        # Overwrite batch size to full size of attack dataset
        batch_size = len(self.attack_dataset.df)
        return DataLoader(
            self.attack_dataset, batch_size, drop_last=True, shuffle=False
        )


class MembershipInferenceModule(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 4),
            nn.GELU(),
            nn.Linear(4, 1),
            # nn.Sigmoid(), # using BCELosswithlogits - no sigmoid needed
        )

    def forward(self, x):
        outputs = self.layers(x)
        return outputs


class MembershipInferenceModel(pl.LightningModule):
    def __init__(self, learning_rate, input_size):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model = MembershipInferenceModule(input_size)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim

    def forward(self, kwh):
        model_inputs = kwh
        return self.model(model_inputs)

    def predict(self, kwh, return_raw_prob=False):
        logits = self.forward(kwh)
        probs = torch.sigmoid(logits)

        if return_raw_prob:
            return probs
        else:
            return probs.round()

    def calculate_accuracy(self, pred, label):
        return torchmetrics.functional.classification.binary_accuracy(
            pred, label
        )

    def training_step(self, batch_data):
        tensors = batch_data[0]
        label = batch_data[1]

        logit = self.forward(tensors)
        pred = self.predict(tensors)

        loss = self.loss_fn(logit.squeeze(), label.squeeze().float())
        accuracy = self.calculate_accuracy(pred.squeeze(), label.squeeze())

        self.log(
            "Train BCE Loss", loss, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log(
            "Train Accuracy",
            accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch_data, batch_idx):
        tensors = batch_data[0]
        label = batch_data[1]

        logit = self.forward(tensors)
        pred = self.predict(tensors)

        loss = self.loss_fn(logit.squeeze(), label.squeeze().float())
        accuracy = self.calculate_accuracy(pred.squeeze(), label.squeeze())

        self.log(
            "Eval BCE Loss", loss, on_epoch=True, on_step=False, prog_bar=True
        )
        self.log(
            "Eval Accuracy",
            accuracy,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss


def get_mia_predictions(
    model: MembershipInferenceModel, attack_dm: MembershipInferenceDataModule
):
    model.eval()
    dl = attack_dm.attack_dataloader()
    attack_kwh, attack_label = next(iter(dl))

    # Get predictions - probability and label
    pred_proba = model.predict(attack_kwh, return_raw_prob=True)
    pred_label = model.predict(attack_kwh, return_raw_prob=False)

    # Safe results to dataframe in corresponding records
    df_results = attack_dm.attack_dataset.df.copy()
    df_results["pred_proba"] = pred_proba.squeeze().detach().numpy()
    df_results["pred_label"] = pred_label.squeeze().detach().numpy()
    df_results["true_label"] = attack_label.detach().numpy()

    # Rank predictions
    df_results = df_results.sort_values("pred_proba", ascending=False)
    ones = df_results["true_label"].sum() * [1]
    zeros = (len(df_results) - len(ones)) * [0]
    df_results["rank_pred"] = ones + zeros
    df_results["pred_label"] = df_results["pred_label"].astype(int)
    df_results["true_label"] = df_results["true_label"].astype(int)
    df_results["rank_pred"] = df_results["rank_pred"].astype(int)

    return df_results


def print_mia_results(df: pd.DataFrame):
    df["accuracy"] = df["true_label"] == df["rank_pred"]

    precision = precision_score(df["true_label"], df["rank_pred"])
    f1 = f1_score(df["true_label"], df["rank_pred"])
    acc = accuracy_score(df["true_label"], df["rank_pred"])
    true_positive = len(df.query("true_label==1 and rank_pred==1"))
    false_positive = len(df.query("true_label==0 and rank_pred==1"))
    total_n = len(df)
    total_positive = df["true_label"].sum()

    interval = 1.96 * np.sqrt((precision * (1 - precision)) / total_n)

    print(
        f"""Segment: Precision: {precision:.2f} +/- {interval:.2f},"""
        f"""\nF1: {f1:.2f}, Acc: {acc:.2f},"""
        f"""\nTrue Positives: {true_positive},"""
        f"""\nFalse Positive: {false_positive},"""
        f"""\nTotal N: {total_n},"""
        f"""\nTotal Positive: {total_positive},"""
        f"""\nRandom Precision : {total_positive/total_n:.2f},"""
    )
