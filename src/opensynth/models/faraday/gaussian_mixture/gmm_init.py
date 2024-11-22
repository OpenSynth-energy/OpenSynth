# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.gaussian_mixture import gmm_model, gmm_utils
from opensynth.models.faraday.vae_model import FaradayVAE


def initialise_gmm_params(
    data: TrainingData,
    n_components: int,
    vae_module: FaradayVAE,
    reg_covar: float = 1e-6,
    sample_weights_column: Optional[str] = None,
) -> gmm_model.GMMInitParams:
    """
    Initialise Gaussian Mixture Parameters.
    Component means and component assignments are initialised using K-means,
      and porting SKLearn's implementation of computing cholesky precision and
      covariances.

    Args:
        data (TrainingData): Input data
        n_components (int): Number of components
        vae_module(FaradayVAE): Trained VAE model
        reg_covar (float): Regularisation for covariance matrix

    Returns:
        dict[str, torch.Tensor]: GMM params
    """

    encoded_data = gmm_utils.prepare_data_for_training_step(
        data, vae_module, sample_weights_column
    )
    X = encoded_data.detach().numpy()

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
