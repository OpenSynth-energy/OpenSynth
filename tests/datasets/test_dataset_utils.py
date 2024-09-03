import numpy as np
import pytest

from opensynth.datasets.datasets_utils import NoiseFactory, NoiseType
from tests.utils import df_test


class TestNoiseFactory:
    df_test = df_test()

    @pytest.mark.parametrize(
        "noise_type",
        [
            pytest.param(NoiseType.GAUSSIAN),
            pytest.param(NoiseType.GAMMA),
            pytest.param(
                "bad_noise",
                marks=pytest.mark.xfail(raises=ValueError, strict=True),
            ),
        ],
    )
    def test_noise_type(self, noise_type):
        test_generator = NoiseFactory(
            noise_type=noise_type, mean=1, scale=1, mean_factor=1, size=(1, 1)
        )
        test_generator.generate_noise()

    @pytest.mark.parametrize(
        "noise_dim",
        [
            pytest.param(48),
            pytest.param(96),
        ],
    )
    def test_df_noise_dimensions(self, noise_dim):
        test_generator = NoiseFactory(
            noise_type=NoiseType.GAUSSIAN,
            mean=1,
            scale=1,
            mean_factor=1,
            size=(len(self.df_test), noise_dim),
        )
        test_noise_df = test_generator.inject_noise(self.df_test)
        noise_array = np.array(test_noise_df["kwh"].tolist())
        assert noise_array.shape == (len(self.df_test), noise_dim)

    @pytest.mark.parametrize(
        "noise_type,mean,mean_factor",
        [
            pytest.param(NoiseType.GAMMA, 1, 10),
            pytest.param(NoiseType.GAMMA, 1, 20),
        ],
    )
    def test_noise_distribution(self, noise_type, mean, mean_factor):
        test_generator = NoiseFactory(
            noise_type=noise_type,
            mean=mean,
            scale=1,
            mean_factor=mean_factor,
            size=(100, 100),
        )
        test_noise = test_generator.generate_noise()
        test_mean = test_noise.mean()
        assert np.rint(test_mean) == np.rint(mean * mean_factor)
