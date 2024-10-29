import numpy as np

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday import FaradayVAE
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    encode_data_for_gmm,
)
from opensynth.models.faraday.new_gmm import gmm_utils, new_gmm_model


def initialise_gmm_params(
    X: np.array, n_components: int
) -> new_gmm_model.GMMInitParams:
    """
    Initialise Gaussian Mixture Parameters. This works
    by only initialising on the first batch of the data
    using K-means, and porting SKLearn's implementation
    of computing cholesky precision and covariances.

    Args:
        X (np.array): Input data
        n_components (int): Number of components

    Returns:
        dict[str, torch.Tensor]: GMM params
    """
    labels_, means_, responsibilities_ = gmm_utils.initialise_centroids(
        X=X, n_components=n_components
    )

    weights_, means_, covariances_ = (
        gmm_utils.torch_estimate_gaussian_parameters(
            X=X,
            means=means_,
            responsibilities=responsibilities_,
            reg_covar=1e-6,
        )
    )

    precision_cholesky_ = gmm_utils.torch_compute_precision_cholesky(
        covariances=covariances_
    )

    init_params = new_gmm_model.GMMInitParams(
        init_labels=labels_,
        means=means_,
        weights=weights_,
        covariances=covariances_,
        precision_cholesky=precision_cholesky_,
    )
    return init_params


def train_gmm(dm: LCLDataModule, vae_module: FaradayVAE, n_components: int):

    # Initialise GMM Params
    first_batch = next(iter(dm.train_dataloader()))
    vae_module.eval()
    input_data = (
        encode_data_for_gmm(data=first_batch, vae_module=vae_module)
        .detach()
        .numpy()
    )
    init_params = initialise_gmm_params(
        X=input_data,
        n_components=n_components,
    )

    # Create GMM Module
    gmm_module = new_gmm_model.GaussianMixtureModel(
        num_components=n_components,
        num_features=input_data.shape[1],
    )
    gmm_module.initialise(init_params=init_params)

    return init_params
