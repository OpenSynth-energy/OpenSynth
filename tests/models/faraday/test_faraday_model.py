import numpy as np
import pytest
import torch

from opensynth.data_modules.lcl_data_module import TrainingData
from opensynth.models.faraday.model import FaradayModel
from opensynth.models.faraday.vae_model import FaradayVAE


@pytest.fixture
def kwh_tensor() -> torch.Tensor:
    kwh = np.random.rand(100, 48)
    return torch.from_numpy(kwh).float()


@pytest.fixture
def feature_dict() -> dict[str, torch.Tensor]:
    feature_list = ["feature_1", "feature_2"]
    output_dict: dict[str, torch.Tensor] = {}
    for feature in feature_list:
        # numpy randint high is exclusive!
        random_tensor = np.random.randint(low=0, high=6, size=100)
        output_dict[feature] = torch.from_numpy(random_tensor)
    return output_dict


def test_faraday_vae_reshape_data(kwh_tensor, feature_dict):
    reshaped_data = FaradayVAE.reshape_data(kwh_tensor, feature_dict)
    expected_dim1 = len(kwh_tensor)
    expected_dim2 = kwh_tensor.shape[1] + len(feature_dict)
    assert reshaped_data.shape == (expected_dim1, expected_dim2)


def test_faraday_model_feature_range(feature_dict):
    got_range = FaradayModel.get_feature_range(feature_dict)
    assert got_range == {
        "feature_1": {"min": 0, "max": 5},
        "feature_2": {"min": 0, "max": 5},
    }


def test_get_index(feature_dict):
    feature_list = list(feature_dict.keys())
    for feature_index, feature_value in enumerate(feature_list):
        if feature_value == "feature_1":
            # There are two items in the list, so first item
            # should have index = -2 [last X column]
            assert FaradayModel.get_index(feature_list, feature_index) == -2
        elif feature_value == "feature_2":
            # There are two items in the list, so second item
            # should have index = -1 [last column]
            assert FaradayModel.get_index(feature_list, feature_index) == -1


class TestFaradayModelParseLabelsAndProfiles:

    samples = torch.rand(12, 12)
    latent_size = 10
    feature_dict = {
        "feature_1": torch.tensor([1, 2, 3, 4, 5]),
        "feature_2": torch.tensor([1, 2, 3, 4, 5]),
    }
    parsed_samples = FaradayModel.parse_samples(
        samples=samples,
        latent_dim=latent_size,
        feature_list=feature_dict.keys(),
    )

    parsed_kwh = parsed_samples["kwh"]
    parsed_features = parsed_samples["features"]

    def test_parsed_features_have_the_right_length(self):
        # Given a set of samples of size 12X12 and a latent size 10
        # We should expect there are 2 features being extracted
        assert (
            len(self.parsed_features)
            == self.samples.shape[1] - self.latent_size
        )

    def test_parsed_kwh_has_the_right_shape(self):
        # Given a set of samples of size 12X12 and a latent size 10
        # We should expect kWh tensor to have shape 12X10
        assert self.parsed_kwh.shape == (
            self.samples.shape[0],
            self.latent_size,
        )

    def test_parsed_features_has_the_right_shape(self):
        # Each of the parsed features should have shape 12X1
        for feature in self.feature_dict:
            assert self.parsed_features[feature].shape == (
                self.samples.shape[0],
                1,
            )

    def test_parsed_samples_are_torch_tensors(self):
        assert isinstance(self.parsed_kwh, torch.Tensor)
        for feature in self.parsed_features:
            assert isinstance(self.parsed_features[feature], torch.Tensor)


def test_faraday_model_get_mask(feature_dict):

    test_labels = np.array([100] * 100)

    # Calculate the mask explicitly
    feature_1_mask = (
        test_labels <= feature_dict["feature_1"].detach().numpy().max()
    )
    feature_2_mask = (
        test_labels <= feature_dict["feature_2"].detach().numpy().max()
    )
    expected_mask = feature_1_mask & feature_2_mask

    # Compare expected mask vs results from create_mask
    test_dict: dict[str, torch.Tensor] = {
        "feature_1": torch.from_numpy(test_labels),
        "feature_2": torch.from_numpy(test_labels),
    }
    test_range = FaradayModel.get_feature_range(feature_dict)
    got_mask = FaradayModel.create_mask(test_dict, test_range)
    assert len(got_mask) == len(test_labels)
    assert (got_mask == expected_mask).all()


def test_filter_mask():

    mask = np.array([True, False, True, False, True])
    test_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    test_samples = TrainingData(
        kwh=test_tensor,
        features={
            "feature_1": test_tensor,
            "feature_2": test_tensor,
        },
    )
    expected_tensor = torch.tensor([1.0, 3.0, 5.0])
    got_samples = FaradayModel.filter_mask(mask, test_samples)

    got_kwh = got_samples["kwh"]
    got_features = got_samples["features"]
    assert torch.equal(got_kwh, expected_tensor)
    for feature in got_features:
        assert torch.equal(got_features[feature], expected_tensor)
