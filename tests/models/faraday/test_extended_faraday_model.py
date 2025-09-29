import numpy as np
import polars as pl
import pytest
import torch

from opensynth.models.faraday import StitchedFaradayModel


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
        for sample in stitched_faraday_model._generate_synthetic_samples(
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
        fmt="pandas",
    )
    assert df.shape == (n_samples, 51)


@pytest.fixture
def fake_model():
    """Mock model with two different features."""

    class FakeModel:
        feature_list = ["month", "dayofweek", "total", "is_zero"]
        rng = np.random.default_rng()

        def sample_gmm(self, n):
            features = {
                "month": self.rng.integers(1, 13, n).reshape(-1, 1),
                "dayofweek": self.rng.integers(0, 7, n).reshape(-1, 1),
                "total": self.rng.integers(1, 4, n).reshape(-1, 1),
                "is_zero": self.rng.integers(0, 2, n).reshape(-1, 1),
            }

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

    return StitchedFaradayModel(FakeModel())


@pytest.fixture
def fake_data_module():
    """Data module for FakeModel."""

    class FakeDataModule:
        def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
            return xhat

    return FakeDataModule()


def test_stitching_identical_features(fake_model, fake_data_module):
    """Test if features are consistent between generated samples."""
    result = fake_model.generate_stitched_samples(
        dm=fake_data_module,
        n_samples=10,
        year=2024,
        fmt="polars",
        period="year",
    )
    # The FakeModel returns identical values for each feature set. This means
    # that for all rows, the values should be identical
    assert result.select(pl.exclude("datetime")).unique().shape[0] == 1
