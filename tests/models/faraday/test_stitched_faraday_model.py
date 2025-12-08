from datetime import date

import numpy as np
import polars as pl
import pytest
import torch

from opensynth.models.faraday import StitchedFaradayModel
from opensynth.models.faraday.stitched_model.utils import (
    sample_number_is_sufficient,
)


@pytest.fixture
def dm_mock():
    class DataModuleMock:
        def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
            return (xhat * 10) + 100

    return DataModuleMock()


@pytest.fixture
def stitched_faraday_model():
    from opensynth.models.faraday.vae_model import Decoder  # noqa: F401
    from opensynth.models.faraday.vae_model import Encoder  # noqa: F401

    faraday_model = torch.load(
        "tests/data/evaluation/faraday_model_for_testing", weights_only=False
    )
    model = StitchedFaradayModel(faraday_model)
    return model


@pytest.mark.parametrize("n_samples", (1, 10))
def test_generate_synthetic_samples(
    dm_mock, stitched_faraday_model, n_samples
):
    """Test generation of synthetic samples for one month."""
    test_year = 2024
    test_month = 10

    result = [
        sample
        for sample in stitched_faraday_model._generate_synthetic_daily_samples(
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


@pytest.mark.parametrize("n_samples", (1, 3))
def test_generate_synthetic_sample_df(
    stitched_faraday_model, dm_mock, n_samples
):
    df = stitched_faraday_model._generate_synthetic_sample_df(
        dm=dm_mock,
        n_samples=n_samples,
        year=2024,
        month=1,
    )
    assert df.shape == (n_samples, 51)


class FakeModel:
    rng = np.random.default_rng()

    def __init__(self, change_order=False):
        self.feature_list = (
            [
                "total",
                "dayofweek",
                "is_zero",
                "month",
            ]
            if change_order
            else ["month", "dayofweek", "total", "is_zero"]
        )

    def sample_gmm(self, n):
        features = {
            "month": self.rng.integers(1, 13, n).reshape(-1, 1),
            "dayofweek": self.rng.integers(0, 7, n).reshape(-1, 1),
            "total": self.rng.integers(1, 4, n).reshape(-1, 1),
            "is_zero": self.rng.integers(0, 2, n).reshape(-1, 1),
        }
        features = {f: features[f] for f in self.feature_list}

        kwh = torch.Tensor(
            np.array(
                [
                    np.zeros(48) if is_zero else np.ones(48) * total
                    for is_zero, total in zip(
                        features["is_zero"][:, 0], features["total"][:, 0]
                    )
                ]
            )
        )

        return {
            "kwh": kwh,
            "features": {k: torch.Tensor(v) for k, v in features.items()},
        }


@pytest.fixture
def fake_model():
    """Mock model with two different features."""

    return StitchedFaradayModel(FakeModel())


@pytest.fixture
def fake_model_different_order():
    """Mock model with different feature order."""

    return StitchedFaradayModel(FakeModel(change_order=True))


@pytest.fixture
def fake_data_module():
    """Data module for FakeModel."""

    class FakeDataModule:
        def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
            return xhat

    return FakeDataModule()


@pytest.mark.parametrize(
    "model_name", ("fake_model", "fake_model_different_order")
)
def test_stitching_identical_features(request, model_name, fake_data_module):
    """Test if features are consistent between generated samples."""
    model = request.getfixturevalue(model_name)
    result = model.generate_stitched_samples(
        dm=fake_data_module,
        n_samples=10,
        year=2024,
        fmt="polars",
        period="year",
    )

    # The FakeModel returns identical values for each feature set. This means
    # that for all rows, the values should be identical
    assert result.select(pl.exclude("datetime")).unique().shape[0] == 1


def test_sample_number_is_sufficient_valid():
    """Should return True when enough days are present"""
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                start=date(2024, 1, 1), end=date(2024, 1, 31), eager=True
            )
        }
    )
    assert sample_number_is_sufficient(
        df=df, n_samples=1, year=2024, month=1, sampled_features=None
    )
    assert sample_number_is_sufficient(
        df=pl.concat((df, df), how="vertical"),
        n_samples=2,
        year=2024,
        month=1,
        sampled_features=None,
    )


def test_sample_number_is_sufficient_invalid():
    """Should return False when not enough days are present"""
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                start=date(2024, 1, 1), end=date(2024, 1, 31), eager=True
            )
        }
    )
    assert not sample_number_is_sufficient(
        df=df.head(30), n_samples=1, year=2024, month=1, sampled_features=None
    )
    assert not sample_number_is_sufficient(
        df=pl.concat((df, df.head(30)), how="vertical"),
        n_samples=2,
        year=2024,
        month=1,
        sampled_features=None,
    )
