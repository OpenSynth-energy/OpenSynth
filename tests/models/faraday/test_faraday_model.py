import numpy as np
import pytest
import torch

from opensynth.models.faraday import FaradayModel, FaradayVAE


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

    assert (got_mask == expected_mask).all()


def test_get_index(feature_dict: dict[str, torch.Tensor]):
    feature_list = list(feature_dict.keys())
    for feature, value in enumerate(feature_list):
        if feature == "feature_1":
            # First feature in feature_dict should have index = 1
            assert FaradayModel.get_index(feature_list, value) == 1
        elif feature == "feature_2":
            # Second feature in feature_dict should have index = 2
            assert FaradayModel.get_index(feature_list, value) == 2
