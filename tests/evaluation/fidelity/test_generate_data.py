import pandas as pd
import polars as pl
import pytest

from opensynth.datasets.low_carbon_london.load import load_lcl_data_by_year


def test_load_lcl_data_by_year_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_lcl_data_by_year()


@pytest.mark.parametrize(
    "df_fmt,expected_type",
    (("pandas", pd.DataFrame), ("polars", pl.DataFrame)),
)
def test_load_lcl_data_by_year(df_fmt, expected_type):
    """Test loading of a specific year from the LCL data."""
    df = load_lcl_data_by_year(
        fname="tests/data/evaluation/train.csv", year=2012, fmt=df_fmt
    )
    assert df.shape == (480, 32)
    assert "datetime" in df.columns
    assert isinstance(df, expected_type)
