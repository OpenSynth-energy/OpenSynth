# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torch.nn as nn

from src.models.faraday.losses import calculate_training_loss


@dataclass
class TrainingData:
    kwh: torch.tensor
    month: torch.tensor
    dow: torch.tensor


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
            nn.Linear(64, self.latent_dim),
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
            nn.Linear(self.latent_dim, 64),
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

    def forward(self, encoded_x):
        mu = self.mean(encoded_x)
        sigma = self.logvar(encoded_x)
        eps = torch.randn_like(mu)
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
        quantile_lower_weight: float = 0.5,
        quantile_median_weight: float = 4,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.975,
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

        # Save hyperparameters
        self.save_hyperparameters()

        # Layers
        self.encoder = Encoder(self.latent_dim, self.input_dim, self.class_dim)
        self.decoder = Decoder(self.class_dim, self.latent_dim, self.input_dim)
        self.reparametriser = ReparametrisationModule(self.latent_dim)

    def encode(self, input_tensor: torch.tensor):
        encoded_x = self.encoder(input_tensor)
        encoded_x, _, _ = self.reparametriser(encoded_x)
        return encoded_x

    def decode(self, latent_tensor: torch.tensor):
        return self.decoder(latent_tensor)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim

    def forward(self, input_data: TrainingData):
        kwh = input_data.kwh
        month = input_data.month.reshape(len(kwh), 1)
        dow_label = input_data.dow.reshape(len(kwh), 1)

        encoder_inputs = torch.cat([kwh, month, dow_label], dim=1)
        encoder_outputs = self.encoder(encoder_inputs)
        decoder_inputs = torch.cat([encoder_outputs, month, dow_label], dim=1)
        decoder_outputs = self.decoder(decoder_inputs)

        return decoder_outputs

    def training_step(self, batch):
        batch_data = TrainingData(kwh=batch[0], month=batch[1], dow=batch[2])
        vae_outputs = self.forward(batch_data)
        total_loss, mmd_loss, mse_loss, quantile_loss = (
            calculate_training_loss(
                x_hat=vae_outputs,
                x=batch_data.kwh,
                lower_quantile=self.lower_quantile,
                upper_quantile=self.upper_quantile,
                mse_weight=self.mse_weight,
                quantile_upper_weight=self.quantile_upper_weight,
                quantile_lower_weight=self.quantile_lower_weight,
                quantile_median_weight=self.quantile_median_weight,
            )
        )

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
