# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.losses import _expand_samples
from opensynth.models.faraday.vae_model import FaradayVAE


def encode_data_for_gmm(
    data: TrainingData, vae_module: FaradayVAE
) -> torch.Tensor:
    """Prepare data for the GMM by encoding it with the VAE.

    Args:
        data (TrainingData): data for training GMM
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


def expand_weights(
    data_tensor: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
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
    model_input = _expand_samples(data_tensor, weights)
    model_input = model_input[torch.randperm(model_input.size(dim=0))]
    return model_input


def prepare_data_for_model(
    vae_module: FaradayVAE,
    data: TrainingData,
    sample_weight_col: str = "",
) -> torch.Tensor:
    """Prepare data for the GMM by encoding it with the VAE.
     If sample weights are used, then expand the repeat based on the weights
     and shuffle the expanded dataset.

    Args:
        data (TrainingData): data for GMM training.
        vae_module (FaradayVAE): VAE module used for encoding.
        sample_weight_col: Name of sample weight column. Defaults to "".
    Returns:
        torch.Tensor: model inputs consisting of encoded consumption data and
        features.
    """
    model_input = encode_data_for_gmm(data, vae_module)

    if not isinstance(sample_weight_col, str):
        raise TypeError(
            "sample_weight_col should be a string, "
            f"not {type(sample_weight_col)}"
        )

    if sample_weight_col != "":
        try:
            model_input = expand_weights(model_input, data[sample_weight_col])
        except KeyError:
            raise KeyError(f"{sample_weight_col} not found in data")

    return model_input
