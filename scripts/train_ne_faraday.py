# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Full New England Faraday training run (detached, CPU, hours).

Trains the VAE on the full preprocessed training split, then fits a
GMM sweep (n_components 200, 100, 400) over the joint latent+label
space. A checkpoint lands after every stage so partial runs are
usable. Mirrors the smoke-run chain that passed on 2026-07-02.

Run from the repo root:
    nohup pipenv run python scripts/train_ne_faraday.py \
        > /tmp/ne_train_full.log 2>&1 &
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

from opensynth.data_modules.ne_data_module import NEDataModule
from opensynth.datasets.new_england.faraday_ne import NewEnglandFaradayModel
from opensynth.models.faraday.vae_model import FaradayVAE

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("train_ne_faraday")


def report(msg: str) -> None:
    """Progress line that survives whatever the deps do to logging."""
    print(f"[train_ne] {msg}", flush=True)


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

BATCH_SIZE = 1024
VAE_EPOCHS = 150
LATENT_DIM = 16
GMM_SWEEP = [200, 100, 400]  # primary first so a usable model lands early
GMM_MAX_EPOCHS = 100
GMM_TOL = 1e-3
GMM_COVARIANCE_REG = 1e-4


class LossHistory(pl.Callback):
    def __init__(self):
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("total_loss")
        if loss is not None:
            self.epoch_losses.append(float(loss))
            report(
                f"VAE epoch {trainer.current_epoch + 1}/{VAE_EPOCHS} "
                f"total_loss={float(loss):.4f}"
            )


def main(data_dir: Path):
    # Same layout the CLI's --loc convention uses, so checkpoints
    # land where generate-ne-dataset expects them
    train_dir = data_dir / "processed" / "new_england" / "train"
    model_dir = data_dir / "models" / "new_england"

    pl.seed_everything(0)
    model_dir.mkdir(parents=True, exist_ok=True)

    n_available = len(pd.read_csv(train_dir / "data.csv", usecols=["ID"]))
    report(f"Training profiles available: {n_available}")

    t0 = time.time()
    dm = NEDataModule(
        data_path=train_dir / "data.csv",
        stats_path=train_dir / "mean_std.csv",
        batch_size=BATCH_SIZE,
        n_samples=n_available,
        outlier_path=train_dir / "outliers.csv",
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    feature_list = list(batch["features"].keys())
    report(
        f"Data loaded in {time.time() - t0:.0f}s; "
        f"kwh batch {tuple(batch['kwh'].shape)}; features {feature_list}"
    )
    assert batch["kwh"].shape[1] == 96

    # --- VAE ---
    vae = FaradayVAE(
        class_dim=len(feature_list),
        latent_dim=LATENT_DIM,
        input_dim=96,
        learning_rate=1e-3,
        mse_weight=3,
    )
    history = LossHistory()
    trainer = pl.Trainer(
        max_epochs=VAE_EPOCHS,
        accelerator="cpu",
        callbacks=[history],
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )
    t0 = time.time()
    trainer.fit(vae, dm)
    report(f"VAE trained in {(time.time() - t0) / 60:.0f} min")
    assert history.epoch_losses[-1] < history.epoch_losses[0]

    vae_ckpt = model_dir / "ne_vae.ckpt"
    trainer.save_checkpoint(vae_ckpt)
    (model_dir / "ne_vae_meta.json").write_text(
        json.dumps(
            {
                "feature_list": feature_list,
                "epoch_losses": history.epoch_losses,
                "n_training_profiles": n_available,
                "batch_size": BATCH_SIZE,
            },
            indent=2,
        )
    )
    report(f"VAE checkpoint saved: {vae_ckpt}")

    # --- GMM sweep ---
    for k in GMM_SWEEP:
        report(f"=== GMM n_components={k} ===")
        model = NewEnglandFaradayModel(
            vae_module=vae,
            n_components=k,
            tol=GMM_TOL,
            max_epochs=GMM_MAX_EPOCHS,
            covariance_reg=GMM_COVARIANCE_REG,
        )
        t0 = time.time()
        model.train_gmm(dm=dm)
        report(f"GMM k={k} trained in {(time.time() - t0) / 60:.0f} min")

        gmm_ckpt = model_dir / f"ne_gmm_{k}.pt"
        torch.save(
            {
                "n_components": k,
                "gmm_state_dict": model.gmm_module.state_dict(),
                "feature_range": model.feature_range,
                "feature_list": feature_list,
                "covariance_reg": GMM_COVARIANCE_REG,
            },
            gmm_ckpt,
        )
        report(f"GMM checkpoint saved: {gmm_ckpt}")

        # Directional sanity: winter oil-heat cold day vs summer warm
        # day. Use common in-distribution combos: 2018 CT Januarys
        # are mostly temp bins 3-4 and Julys bins 8-9; rarer bins
        # force extrapolation and make this check meaningless.
        winter = model.sample_gmm_conditional(
            {
                "state": 0,
                "archetype": 0,
                "heating_fuel": 2,
                "has_ev": 0,
                "has_pv": 0,
                "month": 1,
                "dayofweek": 2,
                "temp_bin": 3,
            },
            256,
        )
        summer = model.sample_gmm_conditional(
            {
                "state": 0,
                "archetype": 0,
                "heating_fuel": 2,
                "has_ev": 0,
                "has_pv": 0,
                "month": 7,
                "dayofweek": 2,
                "temp_bin": 8,
            },
            256,
        )
        with torch.no_grad():
            w = dm.reconstruct_kwh(winter["kwh"]).clip(min=0).mean()
            s = dm.reconstruct_kwh(summer["kwh"]).clip(min=0).mean()
        report(
            f"Sanity k={k}: winter-cold mean {float(w):.4f} "
            f"vs summer-mild {float(s):.4f} kWh/15min"
        )

    report("ALL STAGES COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory (the CLI's --loc)",
    )
    main(parser.parse_args().data_dir)
