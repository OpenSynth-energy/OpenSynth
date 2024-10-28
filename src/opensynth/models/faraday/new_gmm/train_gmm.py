import torch
from torch.utils.data import DataLoader

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    encode_data_for_gmm,
)
from opensynth.models.faraday.new_gmm import gmm_utils


def initialise_gmm_params(
    data_loader: DataLoader, vae_module: FaradayVAE, n_components: int
) -> dict[str, torch.Tensor]:
    """
    Initialise Gaussian Mixture Parameters. This works
    by only initialising on the first batch of the data
    using K-means, and porting SKLearn's implementation
    of computing cholesky precision and covariances.

    Args:
        data_loader (DataLoader): Data loader
        vae_module (FaradayVAE): VAE Module
        n_components (int): Number of components

    Returns:
        dict[str, torch.Tensor]: GMM params
    """

    first_batch = next(iter(data_loader))

    input_data = (
        encode_data_for_gmm(data=first_batch, vae_module=vae_module)
        .detach()
        .numpy()
    )

    labels_, means_, responsibilities_ = gmm_utils.initialise_centroids(
        dataloader=data_loader,
        vae_module=vae_module,
        n_components=n_components,
    )
    weights_, covariances_ = gmm_utils.torch_estimate_gaussian_parameters(
        X=input_data, means=means_
    )

    init_params: dict[str, torch.Tensor] = {
        "labels": labels_,
        "means": means_,
        "responsibilities": responsibilities_,
        "weights": weights_,
        "covariances": covariances_,
    }

    return init_params


def train_gmm(dm: LCLDataModule, vae_module: FaradayVAE, n_components: int):
    init_params = initialise_gmm_params(
        data_loader=dm.train_dataloader(),
        vae_module=vae_module,
        n_components=n_components,
    )
    return init_params
