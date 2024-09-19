"""
This script fits a Gaussian Mixture Model (GMM) to synthetic data using
Pytorch Lightning. Code is based on the PyCave framework.
"""

import time
from typing import List, Tuple, cast

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.loggers import CSVLogger
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday.gaussian_mixture.fit_kmeans import fit_kmeans
from opensynth.models.faraday.gaussian_mixture.gmm_lightning import (
    GaussianMixtureInitLightningModule,
    GaussianMixtureLightningModule,
)
from opensynth.models.faraday.gaussian_mixture.model import (
    GaussianMixtureModel,
)

logger = CSVLogger("logs", name="gmm_logs")


def fit_gmm(
    data: LCLDataModule,
    num_components: int,
    vae_module: pl.LightningModule,
    num_features: int,
    covariance_type: str = "full",
    gmm_max_epochs: int = 10000,
    gmm_convergence_tolerance: float = 1e-6,
    covariance_regularization: float = 1e-6,
    init_method: str = "kmeans",
    kmeans_max_epochs: int = 500,
    kmeans_convergence_tolerance: float = 1e-4,
    is_batch_training: bool = True,
    accelerator: str = "cpu",
    devices: int = 1,
    logs_dir: str = "logs/gmm_logs",
) -> Tuple[GaussianMixtureLightningModule, pl.Trainer, GaussianMixtureModel]:
    """Fit Gaussian Mixture Model to data using PyTorch Lightning

    Args:
        data (LCLDataModule): training dataset
        num_components (int): number of Gaussian components in the
            mixture model.
        vae_module: pl.LightningModule, A trained VAE model.
        num_features (int): number of features in latent space
            (size of latent space + number of non encoded features)
        covariance_type (str, optional): GMM covariance type.
            Defaults to "full".
        gmm_max_epochs (int, optional): maximum epochs to run GMM fitting.
            Defaults to 10000.
        gmm_convergence_tolerance (float, optional): convergence tolerance for
            early stopping of GMM training. Early stopping happens when the
                negative log probability doesn't change more than this value.
                Defaults to 1e-6.
        covariance_regularization (float, optional): a small value which is
            added to the diagonal of the covariance matrix to ensure that it is
                positive semi-definite. Defaults to 1e-6.
        init_method (str, optional): initialisation method for GMM. Allowed
            "rand" or "kmeans". Defaults to "kmeans".
        kmeans_max_epochs (int, optional): maximum epochs to run k-means
            fitting if init_method = "kmeans". Defaults to 500.
        kmeans_convergence_tolerance (float, optional): convergence tolerance
            for early stopping of k-means training. Defaults to 1e-4.
        is_batch_training (bool, optional): flag whether batch training.
            Defaults to True.
        accelerator (str, optional): accelerator for training.
            Defaults to "cpu".
        devices (int, optional): number of devices (or GPUs) to run training.
            Defaults to 1.
        logs_dir (str, optional): location of the output logs recorded during
            training. Defaults to "logs/gmm_logs".

    Returns:
        GaussianMixtureModel: GMM model
        GaussianMixtureLightningModule: GMM lightning module
        pl.Trainer: Pytorch Lightning Trainer for GMM
    """

    start_time = time.time()

    # Initialize the GMM model
    model_ = GaussianMixtureModel(
        covariance_type, num_components, num_features
    )

    if init_method == "kmeans":
        print("Running K-Means initialisation")
        print("--------------------------------------------------------------")
        # Set initial means for GMM using result from K-means fitting
        centroids = fit_kmeans(
            data,
            num_components,
            vae_module,
            num_features,
            kmeans_max_epochs,
            kmeans_convergence_tolerance,
            accelerator,
            devices,
            plot=False,
        )
        # Use k-means centroids as initial means for GMM
        model_.means.copy_(centroids)
        print("Done K-Means initialisation")
        max_epochs_init = 1
    elif init_method == "rand":
        # Use random initialization for GMM
        max_epochs_init = 1 + int(is_batch_training)

    print("Beginning GMM Training")
    # Initialise GMM model using InitLightningModule
    init_module = GaussianMixtureInitLightningModule(
        model_,
        vae_module,
        num_components=num_components,
        num_features=num_features,
        init_method=init_method,
        covariance_type=covariance_type,
        covariance_regularization=covariance_regularization,
        is_batch_training=is_batch_training,
    )

    pl.Trainer(
        max_epochs=max_epochs_init,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    ).fit(init_module, data)

    # Run GMM fitting
    gmm_module = GaussianMixtureLightningModule(
        model_,  # init_module.model,
        vae_module,
        num_components,
        num_features,
        is_batch_training=is_batch_training,
        convergence_tolerance=gmm_convergence_tolerance,
    )
    trainer = pl.Trainer(
        max_epochs=gmm_max_epochs,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(gmm_module, data)

    delta_time = time.time() - start_time
    print(f"Total training time: {delta_time}")

    if not (numpy.isclose(model_.component_probs.sum(), 1.0)):
        raise (
            ValueError(
                "Gaussian mixture component probabilities do not sum to 1.0"
            )
        )

    return gmm_module, trainer, model_


def extract_samples(
    data: LCLDataModule,
    gmm_module: GaussianMixtureLightningModule,
    trainer,
    out_dir: str,
    n_samples: int = 1000,
) -> torch.Tensor:
    """Extract samples from the GMM

    Args:
        data(LCLDataModule): training data
        model (GaussianMixtureModel): GMM model
        n_samples (int, optional): number of sampled data points to draw.
            Defaults to 500.

    Returns:
        torch.Tensor: samples drawn from the GMM
    """
    # Extract data from the synthetic Gaussian dataset
    samples = gmm_module.model.sample(n_samples)
    # Overplot samples on original data
    x = data.gauss_dataset()
    x_sample = x[torch.randperm(x.size(0))[:n_samples]]

    plot_samples(x_sample, samples, out_dir, density=False)

    loader = DataLoader(
        samples,
        batch_size=len(samples),
    )
    result = trainer.predict(gmm_module, loader)

    result = torch.cat(
        [x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)]
    )
    # Predict cluster assignments
    sample_labels = result.argmax(-1).cpu().detach().numpy()

    df = pd.DataFrame({"x1": samples[:, 0], "label": sample_labels})
    print(df.groupby(["label"]).count())

    perc_bins = numpy.round(
        100 * numpy.bincount(sample_labels) / sample_labels.sum(), 1
    )
    print(f"% of {n_samples} samples in each cluster: {perc_bins}")

    return samples


def plot_samples(
    data: torch.Tensor,
    samples: torch.Tensor,
    out_dir: str,
    f_name: str = "sample_from_distribution",
    density: bool = True,
):
    """Plot the training datapoints and overplot samples drawn from the GMM.
    Use if input_dim == 2.

    Args:
        data (torch.Tensor): _description_
        samples (torch.Tensor): _description_
    """

    if density:
        sns.jointplot(x=samples[:, 0], y=samples[:, 1], kind="kde")
        # plt.savefig(f"{out_dir}/sample_from_distribution.png")
        plt.show()
        plt.close()
    else:
        fig, ax = plt.subplots(1, 1)

        # ax.scatter(data[:, 0], data[:, 1], color="grey", s=0.5, alpha=0.5)
        ax.scatter(samples[:, 0], samples[:, 1], color="red", s=0.5, alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/{f_name}.png")
        plt.show()
        plt.close()


def plot_logs(logs_dir: str, out_dir: str):
    """Plots the negative log likelihood of the latest GMM model run over
        training steps.
    Saves plot to file where logs are stored.

    Args:
        logs_dir (str): location of logs directory. This is where the plot wil
            be saved.
    """
    # Read logs
    metrics = pd.read_csv(f"{out_dir}/metrics.csv")
    # Plot NLL over steps
    plt.plot(metrics["epoch"], metrics["nll"])
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log Likelihood")
    plt.savefig(f"{out_dir}/nll_steps.png")
    # plt.show()
    plt.close()


def sklearn_gmm(data, num_components, out_dir, n_sample=10000):
    # Compare to sklearn
    start_time = time.time()

    X = data.gauss_dataset().cpu().detach().numpy()
    gm = GaussianMixture(
        n_components=num_components, max_iter=1000, tol=1e-6
    ).fit(X)

    delta_time = time.time() - start_time
    print(f"sklearn training time: {delta_time}")

    print(f"SK-LEARN MEANS: {gm.means_}")

    samples = gm.sample(n_sample)[0]

    pred = gm.predict(samples)
    perc_bins = numpy.round(100 * numpy.bincount(pred) / pred.sum(), 1)
    print(r"% of samples in each component:")
    print(perc_bins)

    # Plot a sample of the original dataset
    x_sample = X[numpy.random.randint(X.shape[0], size=n_sample), :]
    # plot_centroids(data, gm.means_)

    plot_samples(
        x_sample, samples, out_dir, f_name="sklean_samples", density=False
    )
