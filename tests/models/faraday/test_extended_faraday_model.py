import pytest
import torch

from opensynth.models.faraday import ExtendedFaradayModel


@pytest.fixture
def dm_mock():
    class DataModuleMock:
        def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
            return (xhat * 10) + 100

    return DataModuleMock()


@pytest.fixture
def extended_faraday_model():
    faraday_model = torch.load(
        "tests/data/evaluation/faraday_model_10.pt", weights_only=False
    )
    model = ExtendedFaradayModel(faraday_model)
    return model


def test_generate_synthetic_samples(dm_mock, extended_faraday_model):
    """Test generation of synthetic samples for one month."""
    n_samples = 10
    test_year = 2024
    test_month = 10

    result = [
        sample
        for sample in extended_faraday_model._generate_synthetic_samples(
            dm=dm_mock,
            year=test_year,
            month=test_month,
            n_samples=n_samples,
        )
    ]

    # test number of samples
    assert len(result) == n_samples

    # test dates
    for full_date, month, day, *_ in result:
        assert month == test_month
        assert 0 <= day <= 30
        assert full_date.month == test_month
        assert full_date.year == test_year


def test_generate_synthetic_sample_df(extended_faraday_model, dm_mock):
    df = extended_faraday_model._generate_synthetic_sample_df(
        dm=dm_mock,
        n_samples=3,
        year=2024,
        month=1,
        fmt="pandas",
    )
    assert df.shape == (3, 51)
