import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.models.faraday.gaussian_mixture.prepare_gmm_input import (
    encode_data_for_gmm,
)
from opensynth.models.faraday.new_gmm import gmm_utils
from opensynth.models.faraday.new_gmm.new_gmm_model import (
    GaussianMixtureLightningModule,
    GaussianMixtureModel,
)
from opensynth.models.faraday.new_gmm.train_gmm import initialise_gmm_params

torch.set_default_dtype(torch.float32)

RANDOM_STATE = 0
torch.manual_seed(RANDOM_STATE)
torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(RANDOM_STATE)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


data_path = Path("../data/processed/historical/train/lcl_data_100K.csv")
stats_path = Path("../data/processed/historical/train/mean_std.csv")
outlier_path = Path("../data/processed/historical/train/outliers.csv")

dm = LCLDataModule(
    data_path=data_path,
    stats_path=stats_path,
    batch_size=100000,
    n_samples=100000,
)
dm.setup()

vae_model = torch.load("../notebooks/faraday/vae_model.pt")
vae_model.eval()

next_batch = next(iter(dm.train_dataloader()))
input_tensor = encode_data_for_gmm(data=next_batch, vae_module=vae_model)
input_data = input_tensor.detach().numpy()
n_samples = len(input_tensor)


N_COMPONENTS = 200
REG_COVAR = 1e-4
EPOCHS = 25
IDX = 0
CONVERGENCE_TOL = 1e-2


labels_, means_, responsibilities_ = gmm_utils.initialise_centroids(
    X=input_data, n_components=N_COMPONENTS
)
print(labels_.dtype, responsibilities_.dtype, means_.dtype)


gmm_init_params = initialise_gmm_params(
    X=input_data,
    n_components=N_COMPONENTS,
    reg_covar=REG_COVAR,
)
print(gmm_init_params["weights"].sum())


class CustomDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataModule(LightningDataModule):
    def __init__(self, data_tensor: torch.Tensor, batch_size: int):
        super().__init__()
        self.data_tensor = data_tensor
        self.batch_size = batch_size

    def setup(self, stage=""):
        self.custom_ds = CustomDataset(self.data_tensor)

    def train_dataloader(self):
        return DataLoader(
            self.custom_ds,
            batch_size=self.batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
        )


custom_dm = CustomDataModule(data_tensor=input_tensor, batch_size=25000)
custom_dm.setup(stage="")

print("\nSTARTING FIT PYTORCH LIGHTNING")
print("------------------------------------------------------------")
gmm_module = GaussianMixtureModel(
    num_components=N_COMPONENTS,
    num_features=input_data.shape[1],
    reg_covar=REG_COVAR,
    print_idx=IDX,
)
gmm_module.initialise(gmm_init_params)
print(
    f"Initial prec chol: {gmm_module.precision_cholesky[IDX][0][0]}. \
        Initial mean: {gmm_module.means[IDX][0]}"
)

gmm_lightning_module = GaussianMixtureLightningModule(
    gmm_module=gmm_module,
    vae_module=vae_model,
    num_components=gmm_module.num_components,
    num_features=gmm_module.num_features,
    reg_covar=gmm_module.reg_covar,
    convergence_tolerance=CONVERGENCE_TOL,
    compute_on_batch=False,
)
trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cpu", deterministic=True)
trainer.fit(gmm_lightning_module, custom_dm)


ligthning_sum_components = gmm_lightning_module.gmm_module.means.sum(axis=1)
n_zeros = len(ligthning_sum_components[ligthning_sum_components == 0])
print(f"Number of zero components: {n_zeros}")

print("\nSTARTING FIT SKLEARN")
print("------------------------------------------------------------")
init_weights = gmm_init_params["weights"]
init_means = gmm_init_params["means"]

skgmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type="full",
    tol=CONVERGENCE_TOL,
    max_iter=EPOCHS,
    random_state=0,
    means_init=init_means,
    weights_init=init_weights,
    warm_start=True,
    verbose=1,
)

dl = custom_dm.train_dataloader()
next_batch = next(iter(dl))
for batch_num, batch_data in enumerate(dl):
    print("Batch number: ", batch_num)
    input_data = batch_data.detach().numpy()
    n_samples = len(input_tensor)
    skgmm.fit(input_data)

sklearn_sum_components = skgmm.means_.sum(axis=1)
n_zeros = len(sklearn_sum_components[sklearn_sum_components == 0])
print(f"Number of zero components: {n_zeros}")


df_compare_means = pd.DataFrame()
df_compare_means["skgmm"] = skgmm.means_[IDX]
df_compare_means["lightning"] = gmm_lightning_module.gmm_module.means[IDX]
print(df_compare_means)
