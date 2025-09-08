import pytest
import torch
import torch.nn as nn

from opensynth.models.faraday import ExtendedFaradayModel


@pytest.fixture
def dm_mock():
    class DataModuleMock:
        def reconstruct_kwh(self, xhat: torch.Tensor) -> torch.Tensor:
            return (xhat * 10) + 100

    return DataModuleMock()


@pytest.fixture
def extended_faraday_model():

    class Encoder(nn.Module):
        def __init__(self, latent_dim: int, input_dim: int, class_dim: int):
            super().__init__()
            self.latent_dim = latent_dim
            self.input_dim = input_dim
            self.class_dim = class_dim
            self.encoder_input_dim = self.input_dim + self.class_dim

            # Encoder layers
            self.encoder_layers = nn.Sequential(
                nn.Linear(self.encoder_input_dim, 512),
                nn.GELU(),
                nn.Linear(512, 32),
                nn.GELU(),
                nn.Linear(32, self.latent_dim),
            )

        def forward(self, x):
            return self.encoder_layers(x)

    class Decoder(nn.Module):
        def __init__(self, class_dim: int, latent_dim: int, output_dim: int):
            super().__init__()
            self.latent_dim = latent_dim
            self.class_dim = class_dim
            self.output_dim = output_dim
            self.decoder_input_dim = self.latent_dim + self.class_dim

            # Layers to map latent space back to FC layers
            self.latent = nn.Linear(self.decoder_input_dim, self.latent_dim)
            self.latent_activations = nn.GELU()

            # Decoder layers
            self.decoder_layers = nn.Sequential(
                nn.Linear(self.latent_dim, 32),
                nn.GELU(),
                nn.Linear(32, 512),
                nn.GELU(),
                nn.Linear(512, self.output_dim),
            )

        def forward(self, x):
            outputs = self.latent(x)
            outputs = self.latent_activations(outputs)
            outputs = self.decoder_layers(outputs)
            return outputs

    faraday_model = torch.load(
        "tests/data/evaluation/faraday_model_for_testing", weights_only=False
    )
    model = ExtendedFaradayModel(faraday_model)
    return model


@pytest.mark.parametrize("n_samples", (1, 10))
def test_generate_synthetic_samples(
    dm_mock, extended_faraday_model, n_samples
):
    """Test generation of synthetic samples for one month."""
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


@pytest.mark.parametrize("n_samples", (1, 3))
def test_generate_synthetic_sample_df(
    extended_faraday_model, dm_mock, n_samples
):
    df = extended_faraday_model._generate_synthetic_sample_df(
        dm=dm_mock,
        n_samples=n_samples,
        year=2024,
        month=1,
        fmt="pandas",
    )
    assert df.shape == (n_samples, 51)
