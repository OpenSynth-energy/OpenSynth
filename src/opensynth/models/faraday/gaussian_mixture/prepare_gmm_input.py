import torch

from opensynth.models.faraday.losses import _expand_samples
from opensynth.models.faraday.model import FaradayVAE


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
        model_input = model_input[
            torch.randperm(model_input.size()[0])
        ]  # Shuffle tensor

    return model_input
