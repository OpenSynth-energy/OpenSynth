# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import torch

from opensynth.models.faraday.gaussian_mixture import gmm_model, gmm_utils
from opensynth.models.faraday.vae_model import FaradayVAE


def initialise_gmm_params(
    data: torch.Tensor,
    n_components: int,
    vae_module: FaradayVAE,
    reg_covar: float = 1e-6,
) -> gmm_model.GMMInitParams:
    """
    Initialise Gaussian Mixture Parameters. This works
    by only initialising on the first batch of the data
    using K-means, and porting SKLearn's implementation
    of computing cholesky precision and covariances.

    Args:
        X (np.array): Input data
        n_components (int): Number of components
        reg_covar (float): Regularisation for covariance matrix

    Returns:
        dict[str, torch.Tensor]: GMM params
    """

    # Use data from first batch to initialise centroids
    # If data is shuffled, this should represent the full dataset
    X = gmm_utils.encode_data(data, vae_module)
    X = X.detach().numpy()

    labels_, means_, responsibilities_ = gmm_utils.initialise_centroids(
        X=X, n_components=n_components
    )

    weights_, means_, covariances_ = (
        gmm_utils.torch_estimate_gaussian_parameters(
            X=torch.from_numpy(X),
            responsibilities=responsibilities_,
            reg_covar=reg_covar,
        )
    )

    precision_cholesky_ = gmm_utils.torch_compute_precision_cholesky(
        covariances=covariances_, reg=reg_covar
    )

    init_params = gmm_model.GMMInitParams(
        init_labels=labels_,
        means=means_,
        weights=weights_,
        covariances=covariances_,
        precision_cholesky=precision_cholesky_,
    )
    return init_params
