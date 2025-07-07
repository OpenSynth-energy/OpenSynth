import pandas as pd
import polars as pl
import pytest

from opensynth.evaluation.fidelity.generate_data import (
    generate_synthetic_samples,
    load_lcl_data_by_year,
)


def test_load_lcl_data_by_year_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_lcl_data_by_year()


@pytest.mark.parametrize(
    "df_fmt,expected_type",
    (("pandas", pd.DataFrame), ("polars", pl.DataFrame)),
)
def test_load_lcl_data_by_year(df_fmt, expected_type):
    df = load_lcl_data_by_year(
        fname="tests/data/evaluation/train.csv", year=2012, fmt=df_fmt
    )
    assert df.shape == (480, 32)
    assert "datetime" in df.columns
    assert isinstance(df, expected_type)


def test_generate_synthetic_samples():
    import torch

    from opensynth.data_modules.lcl_data_module import LCLDataModule

    model = torch.load(
        "tests/data/evaluation/faraday_model_1.pt", weights_only=False
    )
    dm = LCLDataModule(
        data_path=".", stats_path=".", batch_size=1, n_samples=1
    )
    result = generate_synthetic_samples(
        model=model,
        dm=dm,
        year=2024,
        month=10,
        n_samples=10,
    )
    print(result)
    assert False
