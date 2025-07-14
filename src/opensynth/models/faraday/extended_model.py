from calendar import monthrange
from collections.abc import Generator
from datetime import date
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm.auto import tqdm

from opensynth.data_modules.lcl_data_module import LCLDataModule
from opensynth.utils.polars import semiwide_to_wide

from .model import FaradayModel

DEFAULT_YEAR = 2024


class ExtendedFaradayModel:
    """Extended Faraday model to generate longer time-series.

    First instantiate an ExtendedFaradayModel instance, where `trained_model`
    is an instance of FaradayModel(), which is already trained.

    >>> model = ExtendedFaradayModel(trained_model)

    Then, use the `generate_extended_samples()` method to generate synthetic samples
    for a month or a year. Keep in mind that Faraday, as currently implemented,
    will not impose any constraints on the consistency between different days.

    >>> generated_samples = model.generate_extended_samples(
            dm=dm,
            month=10,
            n_samples=10,
            year=2023,
            )
    """

    def __init__(self, fm: FaradayModel):

        self.model = fm

    def generate_extended_samples(
        self,
        dm: LCLDataModule,
        n_samples: int,
        month: int | None = None,
        year: int = DEFAULT_YEAR,
        period: Literal["month", "year"] = "month",
        fmt: Literal["pandas", "polars"] = "polars",
    ):
        """Generate DataFrame with Faraday samples for a specific month or year.

        Samples will be generated with all timesteps for all days in the specified
        month or year.

        Args:
            dm (LCLDataModule): Data module.
            n_samples (int): Number of synthetic samples to generate.
            month (int): Specific month, used in combinatoon with `period="month"`.
            year (int, optional): Year to use for timestamps.
            period (str, optional): Generate samples for a full "month", or for a "year".
                Default is "month".
            fmt (str, optional): DataFrame format Either "pandas" or "polars", default is
                "pandas".

        Returns:
            DataFrame in wide format with datetime as first columns.
        """
        match period:
            case "month":
                if month is None:
                    raise ValueError(
                        "The month needs to be specified when period is 'month'"
                    )
                df = self._generate_full_synthetic_month(
                    dm=dm,
                    year=year,
                    month=month,
                    n_samples=n_samples,
                    fmt=fmt,
                )
            case "year":
                df = self._generate_full_synthetic_year(
                    dm=dm, year=year, n_samples=n_samples, fmt=fmt
                )
            case _:
                raise ValueError(
                    "Invalid period, should be either 'month' or 'year'"
                )

        
        
        return df

    def _generate_full_synthetic_month(
        self,
        dm: LCLDataModule,
        year: int,
        month: int,
        n_samples: int = 1,
        fmt: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """Generate DataFrame Faraday samples for a specific month.

        Samples will be generated with a timestamp that fits the specified month.

        Args:
            dm (LCLDataModule): Data module.
            year (int, optional): Year to use for timestamps.
            month (int, optional): Month (1-based) to use. If generated samples do not
                match the specified month, they will be discarded until enough samples
                are specified that do match.
            n_samples (int): Number of synthetic samples to generate.
            fmt (str, optional): Either "pandas" or "polars", default is "pandas".

        Returns:
            pl.DataFrame in wide format with datetime as first columns.
        """
        batch_size = 1000
        df = self._generate_synthetic_sample_df(
            dm, batch_size, year=year, month=month, fmt="polars"
        )

        while (
            df.group_by("date").len().min()["len"][0] < n_samples + 1
            or len(df["date"].unique()) < monthrange(year, month)[1]
        ):
            df = pl.concat(
                (
                    df,
                    self._generate_synthetic_sample_df(
                        dm=dm,
                        n_samples=batch_size,
                        year=year,
                        month=month,
                        fmt="polars",
                    ),
                )
            )
        df = pl.concat(
            [
                p.sample(n_samples + 1).with_row_index()
                for p in df.partition_by("date")
            ]
        )

        
        df = df.filter(pl.col('index') != df['index'].max())
 
        if fmt == "pandas":
            return df.to_pandas()

        return df

    def _generate_full_synthetic_year(
        self,
        dm: LCLDataModule,
        year: int,
        n_samples: int = 1,
        fmt: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """Generate DataFrame Faraday samples for a specific year.

        Samples will be generated with all timesteps for all months in the
        specified year.

        Args:
            dm (LCLDataModule): Data module.
            year (int, optional): Year to use for timestamps.
            n_samples (int): Number of synthetic samples to generate.
            fmt (str, optional): Either "pandas" or "polars", default is "pandas".

        Returns:
            pl.DataFrame in wide format with datetime as first columns.
        """
        df = pl.concat(
            [
                self._generate_full_synthetic_month(
                    dm=dm,
                    year=year,
                    month=month,
                    n_samples=n_samples,
                    fmt="polars",
                )
                for month in tqdm(range(1, 13))
            ]
        )
        df = (
            semiwide_to_wide(
                df.select(pl.exclude("month", "dayofweek")),
                date_col="date",
                datetime_name="datetime",
            )
            .with_columns(pl.col("index").cast(str))
            .transpose(
                column_names="index",
                include_header=True,
                header_name="datetime",
            )
            .with_columns(pl.col("datetime").str.to_datetime())
        )

        if fmt == "pandas":
            return df.to_pandas()

        return df

    def _generate_synthetic_samples(
        self,
        dm: LCLDataModule,
        n_samples: int,
        year: int = DEFAULT_YEAR,
        month: int | None = None,
    ) -> Generator[
        Tuple[date, float, float, np.typing.NDArray[np.float64]], None, None
    ]:
        """Generate Faraday samples for a specific month/year combination.

        Samples will be generated with a timestamp that fits the specified
        year and month. If month is not specified, it can be any month.

        Args:
            dm (LCLDataModule): Data module.
            n_samples (int): Number of synthetic samples to generate.
            year (int, optional): Year to use for timestamps.
            month (int, optional): Month (1-based) to use. If generated samples do not
                match the specified month, they will be discarded until enough samples
                are specified that do match.

        Yields:
            Tuple with datetime, month, day_of_week, generated sample values
        """
        if n_samples < 2:
            raise ValueError("n_samples must be higher than 1")

        sample_df = (
            pl.date_range(
                date(year, 1, 1), date(year, 12, 31), "1d", eager=True
            )
            .alias("datetime")
            .to_frame()
            .with_columns(
                pl.col("datetime").dt.weekday().alias("weekday"),
                pl.col("datetime").dt.month().alias("month"),
            )
        )

        n_batch = n_samples * 100
        n_generated = 0
        for _ in range(10):  # 1000 times should be enough
            gmm_samples = self.model.sample_gmm(n_batch)
            gmm_samples_reconstructed = dm.reconstruct_kwh(gmm_samples["kwh"])
            gmm_samples_reconstructed = torch.clip(
                gmm_samples_reconstructed, min=0
            )
            for torch_month, dayofweek, values in zip(
                gmm_samples["features"]["month"],
                gmm_samples["features"]["dayofweek"],
                gmm_samples_reconstructed,
            ):
                g_month = torch_month.numpy()[0]
                try:
                    if month is None or g_month == month:
                        yield (
                            sample_df.filter(
                                pl.col("weekday") == dayofweek.numpy()[0] + 1,
                                pl.col("month") == g_month,
                            ).sample(1)["datetime"][0],
                            g_month,
                            dayofweek.numpy()[0],
                            values.detach().numpy(),
                        )
                        n_generated += 1
                        if n_generated >= n_samples:
                            return

                except Exception as e:
                    print(e)
                    continue

    def _generate_synthetic_sample_df(
        self,
        dm: LCLDataModule,
        n_samples: int,
        year: int = DEFAULT_YEAR,
        month: int | None = None,
        fmt: Literal["pandas", "polars"] = "pandas",
    ) -> pd.DataFrame | pl.DataFrame:
        """Generate DataFrame Faraday samples for a specific month/year combination.

        Samples will be generated with a timestamp that fits the specified year and
        month. If month is not specified, it can be any month.

        Args:
            dm (LCLDataModule): Data module.
            n_samples (int): Number of synthetic samples to generate.
            year (int, optional): Year to use for timestamps.
            month (int, optional): Month (1-based) to use. If generated samples do not
                match the specified month, they will be discarded until enough samples
                are specified that do match.
            fmt (str, optional): Either "pandas" or "polars", default is "pandas".

        Returns:
            pl.DataFrame in wide format with datetime as first columns.
        """
        df = pl.DataFrame(
            np.array(
                [
                    (datetime, m, d, *values)
                    for datetime, m, d, values in self._generate_synthetic_samples(
                        dm, n_samples, year=year, month=month
                    )
                ]
            ).tolist(),
            schema={"date": pl.Date, "month": int, "dayofweek": int}
            | {
                d: float
                for d in [f"{i // 2:02d}{(i % 2) * 30:02d}" for i in range(48)]
            },
            orient="row",
        )

        if fmt == "pandas":
            return df.to_pandas()

        return df
