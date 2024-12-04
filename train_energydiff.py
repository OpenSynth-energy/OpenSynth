import os
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import pandas as pd

from src.opensynth.data_modules.lcl_data_module import LCLDataModule, LCLData
from src.opensynth.models.energydiff import diffusion, model

class LCLDataModuleWithValidation(LCLDataModule):
    def __init__(self, data_path, stats_path, batch_size=32, n_samples=1000, outlier_path=None, n_val_samples=200):
        super().__init__(data_path, stats_path, batch_size, n_samples, outlier_path=outlier_path)
        self.n_val_samples = n_val_samples

    def setup(self, stage=None):
        super().setup(stage)
        self.val_dataset = LCLData(
            data_path=self.data_path,
            stats_path=self.stats_path,
            n_samples=self.n_val_samples,
            outlier_path=self.outlier_path,
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            self.batch_size,
            drop_last=True,
            shuffle=False,
        )
        
def main():
    # prep data
    data_path = Path("data/processed/historical/train/lcl_data.csv")
    stats_path = Path("data/processed/historical/train/mean_std.csv")
    outlier_path = Path("data/processed/historical/train/outliers.csv")

    dm = LCLDataModuleWithValidation(
        data_path=data_path,
        stats_path=stats_path,
        batch_size=200,
        n_samples=20000,
        outlier_path=outlier_path,
    )
    dm.setup()
    
    # prep model
    df_model = diffusion.PLDiffusion1D(
        dim_base=128,
        dim_in=1,
        num_attn_head=4,
        num_decoder_layer=12,
        dim_feedforward=512,
        dropout=0.1,
        learn_variance=False,
        num_timestep=1000,
        model_mean_type=diffusion.ModelMeanType.V,
        model_variance_type=diffusion.ModelVarianceType.FIXED_SMALL,
        loss_type=diffusion.LossType.MSE,
        beta_schedule_type=diffusion.BetaScheduleType.COSINE,
        lr=1e-3,
        ema_update_every=1,
        ema_decay=0.999,
    )
    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_epochs=250,
    )
    
    # training
    trainer.fit(df_model, dm)
    
    # sample
    ema_df_model = df_model.ema.ema_model # GaussianDiffusion1
    synthetic = ema_df_model.dpm_solver_sample(20000, 100, 100, (48, 1))
    log_dir = trainer.logger.log_dir
    os.makedirs(f'{log_dir}/synthetic', exist_ok=True)
    torch.save(synthetic, f'{log_dir}/synthetic/synthetic_dpm.pt')
    
if __name__ == "__main__":
    main()