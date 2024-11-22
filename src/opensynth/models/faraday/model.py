# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch

from opensynth.data_modules.lcl_data_module import LCLDataModule, TrainingData
from opensynth.data_modules.streaming_data_module import StreamDataModule
from opensynth.models.faraday.gaussian_mixture import (
    GaussianMixtureLightningModule,
    GaussianMixtureModel,
    initialise_gmm_params,
)
from opensynth.models.faraday.vae_model import FaradayVAE

logger = logging.getLogger(__name__)


class FaradayModel:

    def __init__(
        self,
        vae_module: FaradayVAE,
        n_components: int,
        tol: float = 1e-3,
        accelerator: str = "cpu",
        devices: int = 1,
        max_epochs: int = 1000,
        covariance_reg: float = 1e-6,
        sample_weights_column: Optional[str] = None,
    ):
        """
        Faraday Model. Note:

        - Faraday Model only supports integer-encoded labels.
        - If you wish to have float labels, you should subclass FaradayModel
        and implement your own `sample_gmm` method.
        - Faraday Model also expects data module to return a
        TrainingData object.

        Args:
            vae_module (FaradayVAE): Trained VAE component.
            n_components (int): GMM clusters.
            tol (float, optional): Tolerance for GMM. Defaults to 1e-3.
            accelerator (str, optional): Accelerator for GMM training.
                Defaults to "cpu".
            devices (int, optional): Number of devices for GMM training.
                Defaults to 1.
            max_epochs (int, optional): Max epochs for GMM training.
                Defaults to 1000.
            covariance_reg (float, optional): Covariance
                regularization for GMM. Defaults to 1e-6.
                This is added to the diagonal of the covariance matrix to
                ensure that it is positive semi-definite. Higher values will
                make the algorithm more robust to singular covariance matrices,
                at the cost of higher regularization.
        """
        self.n_components = n_components
        self.vae_module = vae_module
        self.tol = tol
        self.accelerator = accelerator
        self.devices = devices
        self.max_epochs = max_epochs
        self.covariance_reg = covariance_reg
        self.sample_weights_column = sample_weights_column

    @staticmethod
    def parse_samples(
        samples: torch.Tensor, latent_dim: int, feature_list: list[str]
    ) -> TrainingData:
        """
        Abstract the labels and round numerical values to integers
        Order of features needs to be preserved so VAE can decode
        it correctly, and is handled by `get_index` method.

        Args:
            samples (torch.Tensor): Samples sampled from GMM
            latent_dim (int): Latent Dimension Size
            feature_list (list[str]): List of feature names

        Returns:
            TrainingData: TypedDict containing GMM sampled
            kWh and feature labels
        """

        kwh: torch.Tensor = samples[:, :latent_dim]
        labels: dict[str, torch.Tensor] = {}

        for i, feature in enumerate(feature_list):
            index = FaradayModel.get_index(feature_list, i)
            feature_tensor = torch.round(
                torch.Tensor(samples[:, index]), decimals=0
            ).int()
            labels[feature] = feature_tensor.reshape(len(kwh), 1)

        return TrainingData(kwh=kwh, features=labels)

    @staticmethod
    def get_feature_range(
        features: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, int]]:
        """
        Get the max and min values of numerically encoded features
        Args:
            features (dict[str, torch.Tensor]): Dictionary of
            feature tensors
        Returns:
            dict[str, dict[str, int]]: A dictionary of
            min and max values for each feature label
        """
        feature_range: dict[str, dict[str, int]] = {}
        for feature in features:
            feature_range[feature] = {
                "min": features[feature].min().item(),
                "max": features[feature].max().item(),
            }
        return feature_range

    @staticmethod
    def create_mask(
        gmm_labels: dict[str, torch.Tensor],
        range_dict: dict[str, dict[str, int]],
    ) -> npt.NDArray[np.bool_]:
        """
        Create a filter mask to make sure that GMM sampled labels are within
        the range of accepted values. There is no guarantee that GMM will not
        create out-of-distribution values. For example if `month of year`
        ranges from 0-11, there is no guarantee that GMM will not
        result in `month of year` = 100.
        Args:
            gmm_labels (dict[str, torch.Tensor]): Dictionary of features
            obtained from GMM model
            range_dict (dict[str, dict[str, int]]): Dictionary of
            ranges of each label
        Returns:
            list[bool]: Mask to filter out-of-distribution values
        """
        label_mask: Optional[npt.NDArray[np.bool_]] = None  # Initialize mask

        for feature, bounds in range_dict.items():

            # Fetch the feature range
            min_value = bounds.get("min")
            max_value = bounds.get("max")
            if min_value is None or max_value is None:
                raise ValueError(f"Feature {feature} have missing range")

            # Dynamically fetch the respective labels
            gmm_value: torch.Tensor = gmm_labels.get(feature)
            if gmm_value is None:
                raise ValueError(f"Feature {feature} not found in GMM labels")

            # Create the mask for this label
            current_mask = np.logical_and(
                gmm_value.detach().numpy() >= min_value,
                gmm_value.detach().numpy() <= max_value,
            )

            # Combine the masks with `&` (AND operation)
            label_mask = (
                current_mask
                if label_mask is None
                else (label_mask & current_mask)
            )

        label_mask = np.squeeze(label_mask)
        return np.squeeze(label_mask)

    @staticmethod
    def get_index(feature_list: list[str], current_index: int) -> int:
        """
        Get the index of each label, so that we can store the
        correct column as the correct label in the gmm_labels
        dictionary which the VAE needs in order to decode.

        Args:
            feature_list (list[str]): List of features
            current_index (int): Current iter index

        Returns:
            int: Returns colum index
        """
        return -(len(feature_list) - current_index)

    @staticmethod
    def filter_mask(
        mask: npt.NDArray[np.bool_], sampled_data: TrainingData
    ) -> TrainingData:
        """
        Given a mask and GMM sampled data, filter the GMM
        sample with the mask.

        Args:
            mask (npt.NDArray[np.bool_]): Numpy array of boolean values
            sampled_data (TrainingData): GMM sampled data

        Returns:
            TrainingData: Filtered GMM samples
        """
        sampled_data["kwh"] = sampled_data["kwh"][mask]
        for feature in sampled_data["features"]:
            sampled_data["features"][feature] = sampled_data["features"][
                feature
            ][mask]
        return sampled_data

    def train_gmm(self, dm: Union[StreamDataModule, LCLDataModule]):
        """
        Train Gaussian Mixture Module

        Args:
            dm (StreamDataModule, LCLDataModule): Training data
        """
        logger.info("🚀 Training GMM")

        dl = dm.train_dataloader()
        next_batch = next(iter(dl))

        # Explicit check to make sure that data module used
        # to train GMM has features ordered the same way as
        # when training the VAE
        features = next_batch["features"]
        obtained_feature_list = list(features.keys())
        num_features = self.vae_module.latent_dim + len(obtained_feature_list)

        expected_feature_list = self.vae_module.feature_list
        if obtained_feature_list != expected_feature_list:
            logger.error(
                """Feature list required by `vae_module` and does not match
                features specified by the data module.
                """
            )

        # Initialise GMM parameters
        # Initialising on the first batch of the data only
        # If data is shuffled, this should represent the full data distribution
        gmm_init_params = initialise_gmm_params(
            next_batch,
            n_components=self.n_components,
            vae_module=self.vae_module,
            reg_covar=self.covariance_reg,
            sample_weights_column=self.sample_weights_column,
        )

        gmm_module = GaussianMixtureModel(
            num_components=self.n_components,
            num_features=num_features,
            reg_covar=self.covariance_reg,
        )
        gmm_module.initialise(gmm_init_params)
        print(
            f"Initial prec chol: {gmm_module.precision_cholesky[0][0][0]}. \
                Initial mean: {gmm_module.means[0][0]}"
        )

        # Fit GMM
        gmm_lightning_module = GaussianMixtureLightningModule(
            gmm_module=gmm_module,
            vae_module=self.vae_module,
            num_components=gmm_module.num_components,
            num_features=gmm_module.num_features,
            reg_covar=gmm_module.reg_covar,
            convergence_tolerance=self.tol,
            sync_on_batch=False,
            sample_weights_column=self.sample_weights_column,
        )
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="cpu",
            deterministic=True,
        )
        trainer.fit(gmm_lightning_module, dl)

        # Record the ranges of features seen during training
        # Assuming that batches are random, than this should
        # be representative.
        self.feature_range = self.get_feature_range(features)
        logger.info("🎉 GMM Training Completed")

        self.gmm_module = gmm_module

    def sample_gmm(self, n_samples: int) -> TrainingData:
        """
        Samples latent codes from GMM and decode with decoder.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            TrainingData: Decoder output (KWH), feature labels
        """

        gmm_samples: torch.Tensor = self.gmm_module.sample(n_samples)

        # Parse GMM samples
        gmm_samples_parsed: TrainingData = self.parse_samples(
            gmm_samples,
            self.vae_module.latent_dim,
            self.vae_module.feature_list,
        )

        # Filter invalid (out of distribution) samples
        label_mask = self.create_mask(
            gmm_samples_parsed["features"], self.feature_range
        )
        gmm_samples_filtered = self.filter_mask(label_mask, gmm_samples_parsed)

        # Decode samples with VAE
        decoder_input = self.vae_module.reshape_data(
            gmm_samples_filtered["kwh"], gmm_samples_filtered["features"]
        ).float()

        decoder_output = self.vae_module.decode(decoder_input)

        # Output samples
        outputs = TrainingData(
            kwh=decoder_output, features=gmm_samples_filtered["features"]
        )
        return outputs
