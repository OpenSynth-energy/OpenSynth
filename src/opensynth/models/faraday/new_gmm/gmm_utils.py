from typing import Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


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
        return labels, means, responsibilities
    else:
        raise NotImplementedError("Only kmeans is supported for now")


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

    n_components, n_features = means.shape
    covariances = torch.empty((n_components, n_features, n_features))
    # Avoid division by zero error
    means_eps = means + torch.finfo(means.dtype).eps

    for k in range(n_components):
        diff = X - means_eps[k]
        covariances[k] = (
            torch.matmul(responsibilities[:, k] * diff.T, diff) / weights[k]
        )

    # Add small regularisation
    covariances += reg_covar
    weights = weights / weights.sum()
    return weights, means, covariances


def torch_compute_precision_cholesky(
    covariances: torch.Tensor,
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

    n_components, n_features, _ = covariances.shape
    precisions_chol = torch.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = torch.linalg.cholesky(covariance, upper=False)
        except torch.linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = torch.linalg.solve_triangular(
            cov_chol, torch.eye(n_features), upper=False
        ).T
    return precisions_chol
