from pathlib import Path

import pytorch_lightning as pl

from src.opensynth.data_modules.lcl_data_module import LCLDataModule
from src.opensynth.models.energydiff import diffusion, model

def main():
    # prep data
    data_path = Path("data/processed/historical/train/lcl_data.csv")
    stats_path = Path("data/processed/historical/train/mean_std.csv")
    outlier_path = Path("data/processed/historical/train/outliers.csv")
    
    dm = LCLDataModule(
        data_path=data_path,
        stats_path=stats_path,
        batch_size=200,
        n_samples=2000,
    )
    dm.setup()
    
    # prep model
    denoise_model = model.DenoisingTransformer(
        dim_base=256,
        dim_in=1,
        num_attn_head=4,
        num_decoder_layer=6,
        dim_feedforward=1024,
        dropout=0.1,
        learn_variance=False,
    )
    df_model = diffusion.PLDiffusion1D(
        base_model=denoise_model,
        num_timestep=1000,
        model_mean_type=diffusion.ModelMeanType.V,
        model_variance_type=diffusion.ModelVarianceType.FIXED_SMALL,
        loss_type=diffusion.LossType.MSE,
        beta_schedule_type=diffusion.BetaScheduleType.COSINE,
        lr=1e-4,
        ema_update_every=5,
        ema_decay=0.999,
    )
    trainer = pl.Trainer(
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        max_epochs=10,
    )
    trainer.fit(df_model, dm)
    
if __name__ == "__main__":
    main()