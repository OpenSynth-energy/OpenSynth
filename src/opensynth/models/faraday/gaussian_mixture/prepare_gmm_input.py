# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
import torch

from opensynth.models.faraday.losses import _expand_samples
from opensynth.models.faraday.vae_model import FaradayVAE


def prepare_data_for_model(
    vae_module: FaradayVAE, data: torch.Tensor
) -> torch.Tensor:
    """Prepare data for the GMM by encoding it with the VAE.
     If sample weights are used, then expand the repeat based on the weights
     and shuffle the expanded dataset.

    Args:
        data (torch.Tensor): data for GMM training.
        vae_module (FaradayVAE): VAE module used for encoding.
    Returns:
        torch.Tensor: model inputs consisting of encoded consumption data and
        features.
    """
    kwh = data["kwh"]
    features = data["features"]
    vae_input = vae_module.reshape_data(kwh, features)
    vae_output = vae_module.encode(vae_input)
    model_input = vae_module.reshape_data(vae_output, features)
    if "weights" in data:
        weights = data["weights"]
        model_input = _expand_samples(model_input, weights)
        # Shuffle tensor to prevent consecutive samples from being the same
        # after expansion.
        model_input = model_input[torch.randperm(model_input.size(dim=0))]

    return model_input
