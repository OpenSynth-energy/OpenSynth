from pathlib import Path

from src.opensynth.data_modules.lcl_data_module import LCLDataModule


def main():
    data_path = Path("data/processed/historical/train/lcl_data.csv")
    stats_path = Path("data/processed/historical/train/mean_std.csv")
    outlier_path = Path("data/processed/historical/train/outliers.csv")
    dm = LCLDataModule(
        data_path=data_path,
        stats_path=stats_path,
        batch_size=200,
        n_samples=2000,
    )  # next(iter(dm.train_dataloader()))['kwh'].shape
    # -> torch.Size([batch_size, 48])
    dm.setup()
    dm_with_outliers = LCLDataModule(
        data_path=data_path,
        stats_path=stats_path,
        batch_size=100,
        n_samples=1000,
        outlier_path=outlier_path,
    )
    dm_with_outliers.setup()
    pass


if __name__ == "__main__":
    main()
