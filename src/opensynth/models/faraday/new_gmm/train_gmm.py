import numpy as np
import torch
from tqdm import tqdm

from opensynth.models.faraday.new_gmm import gmm_utils, new_gmm_model


def initialise_gmm_params(
    X: np.array, n_components: int, reg_covar=1e-6, method: str = "torch"
) -> new_gmm_model.GMMInitParams:
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

    init_params = new_gmm_model.GMMInitParams(
        init_labels=labels_,
        means=means_,
        weights=weights_,
        covariances=covariances_,
        precision_cholesky=precision_cholesky_,
    )
    return init_params


def training_loop(
    model: new_gmm_model.GaussianMixtureModel,
    # vae_module: FaradayVAE,
    data: torch.Tensor,
    max_iter: int,
    convergence_tol: float = 1e-2,
):
    converged = False
    lower_bound = -np.inf

    # encoded_batch = encode_data_for_gmm(data=data, vae_module=vae_module)
    for i in tqdm(range(max_iter)):
        prev_lower_bound = lower_bound

        log_prob, log_resp = model.e_step(data)
        precision_cholesky, weights, means, covariances = model.m_step(
            data, log_resp
        )
        model.update_params(
            weights=weights,
            means=means,
            precision_cholesky=precision_cholesky,
            covariances=covariances,
        )
        # Converegence
        lower_bound = log_prob
        change = abs(lower_bound - prev_lower_bound)
        print(f"Change: {change}")
        if change < convergence_tol:
            converged = True
            break

    print(f"Converged: {converged}. Number of iterations: {i}")

    return model
