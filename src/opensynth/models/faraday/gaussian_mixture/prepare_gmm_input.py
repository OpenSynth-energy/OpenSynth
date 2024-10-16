# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
import torch

from opensynth.models.faraday.losses import _expand_samples
from opensynth.models.faraday.vae_model import FaradayVAE


def encode_data_for_gmm(
    data: torch.Tensor, vae_module: FaradayVAE
) -> torch.Tensor:
    """Prepare data for the GMM by encoding it with the VAE.

    Args:
        data (torch.Tensor): data for training GMM
        vae_module (FaradayVAE): trained VAE model

    Returns:
        torch.Tensor: encoded data for GMM
    """
    kwh = data["kwh"]
    features = data["features"]
    vae_input = vae_module.reshape_data(kwh, features)
    vae_output = vae_module.encode(vae_input)
    model_input = vae_module.reshape_data(vae_output, features)
    return model_input


def expand_weights(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Expand the repeat based on the training weights and shuffle the
     expanded dataset.
     Shuffling prevents consecutive samples from being the same to prevent
      convergence issues in GMM training.

    Args:
        data (torch.Tensor): data for training GMM
        weights (torch.Tensor): number of occurances of each data point in the
            dataset

    Returns:
        torch.Tensor:
    """
    model_input = _expand_samples(data, weights)
    model_input = model_input[torch.randperm(model_input.size(dim=0))]
    return model_input


def prepare_data_for_model(
    vae_module: FaradayVAE,
    data: torch.Tensor,
    train_sample_weights: bool,
) -> torch.Tensor:
    """Prepare data for the GMM by encoding it with the VAE.
     If sample weights are used, then expand the repeat based on the weights
     and shuffle the expanded dataset.

    Args:
        data (torch.Tensor): data for GMM training.
        vae_module (FaradayVAE): VAE module used for encoding.
        train_sample_weights: flag whether to train with sample weights.
    Returns:
        torch.Tensor: model inputs consisting of encoded consumption data and
        features.
    """
    model_input = encode_data_for_gmm(data, vae_module)

    if train_sample_weights:
        try:
            model_input = expand_weights(model_input, data["weights"])
        except KeyError:
            raise KeyError(
                f"train_sample_weights set to {train_sample_weights} but"
                "weights not found in data."
            )

    return model_input
