# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

# TODO: Add tests, specifically these functions:
#  get_feature_range, create_mask, get_index
# TODO: Test with Non-LCL data

import logging
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.mixture import GaussianMixture
from torch.optim import lr_scheduler
from tqdm import tqdm

from opensynth.data_modules.lcl_data_module import LCLDataModule, TrainingData
from opensynth.models.faraday.losses import calculate_training_loss

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, class_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.encoder_input_dim = self.input_dim + self.class_dim

        # Encoder layers
        self.encoder_layers = nn.Sequential(
            nn.Linear(self.encoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, self.latent_dim),
        )

    def forward(self, x):
        return self.encoder_layers(x)


class Decoder(nn.Module):
    def __init__(self, class_dim: int, latent_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.output_dim = output_dim
        self.decoder_input_dim = self.latent_dim + self.class_dim

        # Layers to map latent space back to FC layers
        self.latent = nn.Linear(self.decoder_input_dim, self.latent_dim)
        self.latent_activations = nn.GELU()

        # Decoder layers
        self.decoder_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, x):
        outputs = self.latent(x)
        outputs = self.latent_activations(outputs)
        outputs = self.decoder_layers(outputs)
        return outputs


class ReparametrisationModule(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)
        self.norm_dist = torch.distributions.Normal(0, 1)

    def forward(self, encoded_x):
        mu = self.mean(encoded_x)
        sigma = self.logvar(encoded_x)
        eps = self.norm_dist.sample(mu.shape)
        eps = eps.to(encoded_x)
        return mu + eps * sigma, mu, sigma


class FaradayVAE(pl.LightningModule):
    def __init__(
        self,
        class_dim: int,
        learning_rate: float = 1e-3,
        latent_dim: int = 16,
        input_dim: int = 48,
        tmax: int = 5000,
        mse_weight: float = 2.0,
        quantile_upper_weight: float = 4,
        quantile_lower_weight: float = 1,
        quantile_median_weight: float = 4,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        differential_privacy: bool = False,
        epsilon: Optional[float] = 1.0,
        delta: Optional[float] = 1e-5,
        max_grad_norm: Optional[float] = 1.0,
        custom_encoder: Optional[Encoder] = None,
        custom_decoder: Optional[Decoder] = None,
    ):
        super().__init__()
        self.class_dim = class_dim
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.tmax = tmax
        self.learning_rate = learning_rate
        self.mse_weight = mse_weight
        self.quantile_upper_weight = quantile_upper_weight
        self.quantile_lower_weight = quantile_lower_weight
        self.quantile_median_weight = quantile_median_weight
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.differential_privacy = differential_privacy

        if self.differential_privacy:

            if max_grad_norm is None:
                raise ValueError("Max grad norm must be set for DP training")
            elif max_grad_norm > 1 or max_grad_norm < 0:
                raise ValueError("Max grad norm must be between 0 and 1")

            logger.info("🔒 Differential Privacy Enabled")
            logger.info("🔒 Epsilon: {epsilon}, Delta: {delta}")
            logger.info(
                "🔒 Note: to satisfy definition of"
                "differential privacy, Delta: {delta}"
                "must be < 1/N where N is the size of"
                "the training dataset"
            )
            self.max_grad_norm = max_grad_norm
            self.epsilon = epsilon
            self.delta = delta

            self.privacy_engine = PrivacyEngine(secure_mode=False)
            # Check that everything is valid for DP training
            assert ModuleValidator.validate(self, strict=True) == []

        # Save hyperparameters
        self.save_hyperparameters(ignore=["custom_encoder", "custom_decoder"])

        self.encoder = (
            custom_encoder
            if custom_encoder is not None
            else Encoder(self.latent_dim, self.input_dim, self.class_dim)
        )
        self.decoder = (
            custom_decoder
            if custom_decoder is not None
            else Decoder(self.class_dim, self.latent_dim, self.input_dim)
        )
        self.reparametriser = ReparametrisationModule(self.latent_dim)

    def encode(self, input_tensor: torch.tensor):
        encoded_x = self.encoder(input_tensor)
        encoded_x, _, _ = self.reparametriser(encoded_x)
        return encoded_x

    def decode(self, latent_tensor: torch.tensor):
        return self.decoder(latent_tensor)

    def configure_optimizers(self):

        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.differential_privacy:
            self.trainer.fit_loop.setup_data()
            dataloader = self.trainer.train_dataloader
            epochs = self.trainer.max_epochs
            model, optim, dl = self.privacy_engine.make_private_with_epsilon(
                module=self,
                optimizer=optim,
                data_loader=dataloader,
                epochs=epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            self.dp = {"model": model, "optim": optim, "dataloader": dl}

            return optim

        else:
            # Learning Rate schedulers are not compatible with
            # Opacus and Pytorch Lightning
            lr_schedule = lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.tmax, eta_min=self.learning_rate / 5
            )
            return [optim], [lr_schedule]

    def reshape_data(self, kwh_tensor, features: dict[str, torch.tensor]):
        reshaped_batch = torch.cat([kwh_tensor], dim=1)
        for feature in features:
            feature_tensor = features[feature].reshape(len(kwh_tensor), 1)
            reshaped_batch = torch.cat([reshaped_batch, feature_tensor], dim=1)
        return reshaped_batch

    def forward(self, input_data: TrainingData):

        encoder_inputs = self.reshape_data(
            input_data["kwh"], input_data["features"]
        )
        encoder_outputs = self.encode(encoder_inputs)

        decoder_outputs = self.reshape_data(
            encoder_outputs, input_data["features"]
        )
        dec_outputs = self.decode(decoder_outputs)
        return dec_outputs

    def training_step(self, batch: TrainingData):

        vae_outputs = self.forward(batch)
        total_loss, mmd_loss, mse_loss, quantile_loss = (
            calculate_training_loss(
                x_hat=vae_outputs,
                x=batch["kwh"],
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile,
                mse_weight=self.mse_weight,
                quantile_upper_weight=self.quantile_upper_weight,
                quantile_lower_weight=self.quantile_lower_weight,
                quantile_median_weight=self.quantile_median_weight,
            )
        )
        # Save list of feature names
        self.feature_list = list(batch["features"].keys())

        # Might be an overkill to sync_dist for all losses.
        # If this causes significant I/O bottleneck
        # Consider syncing only total_loss
        self.log(
            "total_loss",
            total_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "mmd_loss",
            mmd_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "mse_loss",
            mse_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "quantile_loss",
            quantile_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def on_train_epoch_end(self):
        if self.differential_privacy:
            # Note: Need to convert to float32 because MPS
            # device does not support float64. Will error if you don't convert
            eps = np.float32(self.privacy_engine.get_epsilon(self.delta))
            self.log(
                "epsilon",
                eps,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=False,
            )


class FaradayModel:
    def __init__(
        self,
        vae_module: FaradayVAE,
        n_components: int,
        max_iter: int = 1000,
        covariance_type: str = "full",
        tol: float = 1e-3,
    ):
        """
        Faraday Model

        Args:
            vae_module (FaradayVAE): Trained VAE component
            n_components (int): GMM clusteres
            max_iter (int, optional): Max iteration for GMM. Defaults to 1000.
            covariance_type (str, optional): scikit-learn gmm covariance types.
                Defaults to "full".
            tol (float, optional): Tolerance for GMM. Defaults to 1e-3.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.covariance_type = covariance_type
        self.vae_module = vae_module
        self.tol = tol

        self.gmm = GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            covariance_type=covariance_type,
            warm_start=True,
            tol=tol,
        )

    @staticmethod
    def get_feature_range(features: dict[str, torch.tensor]):
        feature_range: dict[str, dict[str, int]] = {}
        for feature in features:
            feature_range[feature] = {
                "min": features[feature].min().item(),
                "max": features[feature].max().item(),
            }
        return feature_range

    @staticmethod
    def create_mask(gmm_labels, range_dict):
        label_mask = None  # Initialize mask
        for key, bounds in range_dict.items():
            min_value = bounds["min"]
            max_value = bounds["max"]
            # Dynamically fetch the respective labels
            gmm_value = gmm_labels.get(key)
            # Create the mask for this label
            current_mask = (gmm_value >= min_value) & (gmm_value <= max_value)
            # Combine the masks with `&` (AND operation)
            label_mask = (
                current_mask
                if label_mask is None
                else (label_mask & current_mask)
            )
        return label_mask

    @staticmethod
    def get_index(feature_list: list[str], current_index: int):
        return -(len(feature_list) - current_index)

    def train_gmm(self, dm: LCLDataModule):
        """
        Train Gaussian Mixture Module

        Args:
            dm (LCLDataModule): Training data
        """
        dl = dm.train_dataloader()
        next_batch = next(iter(dl))

        # Explicit check to make sure that data module used
        # to train GMM has features ordered the same way as
        # when training the VAE
        obtained_feature_list = list(next_batch["features"].keys())
        expected_feature_list = self.vae_module.feature_list
        assert obtained_feature_list == expected_feature_list

        for batch_num, batch_data in tqdm(enumerate(dl)):
            kwh = batch_data["kwh"]
            features = batch_data["features"]

            vae_input = self.vae_module.reshape_data(kwh, features)
            vae_output = self.vae_module.encode(vae_input)

            gmm_input = self.vae_module.reshape_data(vae_output, features)
            self.gmm.fit(gmm_input.detach().numpy())
            logger.info(f"⏳ Batch {batch_num} completed")

        self.feature_range = self.get_feature_range(features)
        logger.info("🎉 GMM Training Completed")

    def sample_gmm(
        self, n_samples: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples latent codes from GMM and decode with decoder.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor]:
              Decoder output (KWH), month label, dow label
        """
        gmm_samples = self.gmm.sample(n_samples)[0]

        # Parse labels and profiles
        gmm_kwh = gmm_samples[:, : self.vae_module.latent_dim]
        gmm_labels: dict[str, torch.tensor] = {}
        for i, feature in enumerate(self.vae_module.feature_list):
            index = self.get_index(self.vae_module.feature_list, i)
            gmm_labels[feature] = np.round(
                gmm_samples[:, index], decimals=0
            ).astype(int)

        # Filter invalid (out of distribution) samples
        label_mask = self.create_mask(gmm_labels, self.feature_range)
        gmm_kwh = gmm_kwh[label_mask]
        for features in self.feature_range:
            gmm_labels[features] = torch.from_numpy(
                gmm_labels[features][label_mask]
            )

        latent_tensor = torch.from_numpy(gmm_kwh)
        decoder_input = self.vae_module.reshape_data(
            latent_tensor, gmm_labels
        ).float()
        decoder_output = self.vae_module.decode(decoder_input)
        outputs = TrainingData(kwh=decoder_output, features=gmm_labels)
        return outputs
