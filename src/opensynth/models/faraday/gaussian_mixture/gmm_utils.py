# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.losses import _expand_samples
from opensynth.models.faraday.vae_model import FaradayVAE


def _encode_data(data: TrainingData, vae_module: FaradayVAE) -> torch.Tensor:
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


def _expand_weights(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Expand the repeat based on the training weights and shuffle the
     expanded dataset.
     Shuffling prevents consecutive samples from being the same to prevent
      convergence issues in GMM training.

    Args:
        data (torch.Tensor): data for training GMM
        weights (torch.Tensor): number of occurances of each data point in the
            dataset

    Returns:
        torch.Tensor
    """
    model_input = _expand_samples(data, weights)
    model_input = model_input[torch.randperm(model_input.size(dim=0))]
    return model_input


def prepare_data_for_training_step(
    data: TrainingData,
    vae_module: FaradayVAE,
    sample_weights_column: Optional[str] = None,
) -> torch.Tensor:

    encoded_data = _encode_data(data, vae_module)

    if sample_weights_column is not None:
        try:
            model_input = _expand_weights(
                encoded_data, data[sample_weights_column]
            )
        except KeyError:
            raise KeyError(
                f"Column {sample_weights_column} not found in input data."
            )
        return model_input
    else:
        return encoded_data


def initialise_centroids(
    X: np.array,
    n_components: int,
    init_method: str = "kmeans",
    random_state: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialise centroids for GMM.
    Currently only supports K Means.

    Args:
        X (np.array): Input data
        n_components (int): Number of components
        init_method (str, optional): Initialisation method. Defaults
        to "kmeans".
        random_state (int, optional): Random state. Defaults to 0.

    Raises:
        NotImplementedError: if init_method is other than "kmeans"

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns the
        labels, means (centroids), and responsibilities.

        Responsibilities are the one-hot encoded tensor of each sample
        and its labels, of size N X n_components.
    """

    n_samples = len(X)
    dtype = torch.tensor(X).dtype

    if init_method == "kmeans":
        kmeans_model = KMeans(
            n_clusters=n_components, random_state=random_state
        )
        kmeans_model.fit(X)
        labels = torch.from_numpy(kmeans_model.labels_)
        means = torch.from_numpy(kmeans_model.cluster_centers_)

        responsibilities = np.zeros((n_samples, n_components))
        responsibilities[np.arange(n_samples), labels] = 1
        responsibilities = torch.from_numpy(responsibilities)
        return (
            labels.type(dtype),
            means.type(dtype),
            responsibilities.type(dtype),
        )
    else:
        raise NotImplementedError("Only kmeans is supported for now")


def is_symmetric_positive_definite(tensor):
    # Check if the tensor is symmetric
    is_symmetric = all(
        torch.allclose(tensor[i], tensor[i].T, atol=1e-5)
        for i in range(tensor.shape[0])
    )

    # Check if the tensor is positive definite by checking eigenvalues
    eigenvalues = torch.linalg.eigvals(
        tensor
    ).real  # Use only the real part of eigenvalues
    is_positive_definite = torch.all(eigenvalues > 0)

    return is_symmetric and is_positive_definite


def torch_compute_covariance(
    X: torch.Tensor,
    means: torch.Tensor,
    responsibilities: torch.Tensor,
    weights: torch.Tensor,
    reg_covar: float,
) -> torch.Tensor:
    """
    Compute the covariance matrix for each cluster.

    Args:
        X (torch.tensor): Input data
        means (torch.Tensor): Means or centroids
        responsibilities (torch.Tensor): Responsibilities
        weights (torch.Tensor): Weights
        reg_covar (float): Regularisation

    Returns:
        torch.Tensor: Covariance matrix
    """
    n_components, n_features = means.shape
    covariances = torch.empty(
        (n_components, n_features, n_features), device=X.device
    )
    # Avoid division by zero error
    means_eps = means + torch.finfo(means.dtype).eps
    for k in range(n_components):
        diff = X - means_eps[k]
        covariances[k] = (
            torch.matmul(responsibilities[:, k] * diff.T, diff) / weights[k]
        )

        # Add small regularisation
        covariances[k] = (
            covariances[k]
            + torch.eye(n_features, device=covariances.device) * reg_covar
        )

    covariances = covariances.to(device=X.device)

    return covariances


def torch_estimate_gaussian_parameters(
    X: torch.Tensor,
    responsibilities: torch.Tensor,
    reg_covar: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pytorch implmentation of sklearn's
    sklearn.mixture._gaussian_mixture._estimate_gaussian_parameters

    Args:
        X (torch.Tensor): Input data
        responsibilities (torch.Tensor): Reponsibilities,
        i.e. 1-hot encoded tensor of each data and it's cluster label.
        means (torch.Tensor): Coordinate of centroids
        reg_covar (float): Covariance regularisor

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 1) cluster weights
        i.e. number of samples in each cluster and 2) cluster proba:
        % of samples in each cluster 3) covariances

    """
    # n_components, n_features = X.shape
    weights = (
        responsibilities.sum(dim=0) + torch.finfo(responsibilities.dtype).eps
    )
    # Compute new means usint updated responsibilities
    means = torch.matmul(responsibilities.T, X) / weights.reshape(-1, 1)

    covariances = torch_compute_covariance(
        X=X,
        means=means,
        responsibilities=responsibilities,
        weights=weights,
        reg_covar=reg_covar,
    )

    weights = weights / weights.sum()
    return weights, means, covariances


def torch_compute_precision_cholesky(
    covariances: torch.Tensor, reg: float = 1e-6
) -> torch.Tensor:
    """
    Pytorch implmentation of sklearn's
    sklearn.mixture._gaussian_mixture._compute_precision_cholesky
    _compute_precision_cholesky

    Args:
        covariances (torch.Tensor): Covariance matrix

    Raises:
        ValueError: Raises error if matrix is not positive determinate.

    Returns:
        torch.Tensor: Precision Cholesky
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )
    original_device = covariances.device
    if torch.device.type == "mps":
        # torch.linalg.cholesky is not supported on MPS
        # move calculation to CPU and move tensor back to device later
        covariances = covariances.to(torch.device("cpu"))

    n_components, n_features, _ = covariances.shape
    precisions_chol = torch.empty(
        (n_components, n_features, n_features), device=covariances.device
    )

    for k, covariance in enumerate(covariances):
        try:
            cov_chol = torch.linalg.cholesky(covariance, upper=False)
        except torch.linalg.LinAlgError:
            try:
                covariance_fixed = (
                    covariance
                    + torch.eye(n_features, device=covariances.device) * reg
                )
                cov_chol = torch.linalg.cholesky(covariance_fixed, upper=False)
            except torch.linalg.LinAlgError:
                print(f"Failed for {k}th covariance with reg_covar: {reg}.")
                raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = torch.linalg.solve_triangular(
            cov_chol,
            torch.eye(n_features, device=covariances.device),
            upper=False,
        ).T

    precisions_chol = precisions_chol.to(device=original_device)
    return precisions_chol.type(covariances.dtype)
