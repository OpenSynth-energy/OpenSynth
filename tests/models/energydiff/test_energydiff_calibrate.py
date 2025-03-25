import pytest
import torch

from opensynth.models.energydiff.calibrate import (
    DataShapeType,
    MultiDimECDF,
    calibrate,
)


@pytest.fixture
def make_tensor():
    """raw sample data can have whatever distribution"""

    def _make_tensor(shape: DataShapeType):
        if shape == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            return torch.randn(17, 3, 100)
        elif shape == DataShapeType.BATCH_SEQUENCE:
            return torch.randn(17, 300)
        elif shape == DataShapeType.CHANNEL_SEQUENCE:
            return torch.randn(3, 100)
        elif shape == DataShapeType.SEQUENCE:
            return torch.randn(300)
        else:
            raise ValueError("test case not implemented")

    return _make_tensor


@pytest.fixture
def make_uniform_tensor():
    """transformed data should follow U(0, 1) distribution"""

    def _make_uniform(shape: DataShapeType):
        if shape == DataShapeType.BATCH_CHANNEL_SEQUENCE:
            return torch.rand(17, 3, 100)
        elif shape == DataShapeType.BATCH_SEQUENCE:
            return torch.rand(17, 300)
        elif shape == DataShapeType.CHANNEL_SEQUENCE:
            return torch.rand(3, 100)
        elif shape == DataShapeType.SEQUENCE:
            return torch.rand(300)
        else:
            raise ValueError("test case not implemented")

    return _make_uniform


@pytest.fixture(params=list(DataShapeType))
def input_data(request, make_tensor) -> torch.Tensor:
    return make_tensor(request.param)


@pytest.fixture(
    params=[
        DataShapeType.BATCH_CHANNEL_SEQUENCE,
        DataShapeType.BATCH_SEQUENCE,
    ]
)
def ecdf_init_data(request, make_tensor) -> torch.Tensor:
    return make_tensor(request.param)


@pytest.fixture(params=list(DataShapeType))
def uniform_data(request, make_uniform_tensor) -> torch.Tensor:
    return make_uniform_tensor(request.param)


class TestMultiDimECDF:
    @pytest.fixture
    def ecdf_instance(self, ecdf_init_data):
        return MultiDimECDF(ecdf_init_data)

    def test_init(self, ecdf_instance):
        pass

    def test_cdf(self, ecdf_instance):
        x, cdf = ecdf_instance.cdf
        assert isinstance(x, torch.Tensor)
        assert isinstance(cdf, torch.Tensor)
        assert torch.all(cdf >= 0) and torch.all(cdf <= 1)

    def test_transform(self, ecdf_instance, input_data):
        transformed = ecdf_instance.transform(input_data)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == input_data.shape
        assert torch.all(transformed >= 0) and torch.all(transformed <= 1)
        # transformed value follow U(0, 1) distribution

    def test_inverse_transform(self, ecdf_instance, uniform_data):
        inverse = ecdf_instance.inverse_transform(uniform_data)
        assert isinstance(inverse, torch.Tensor)


@pytest.mark.parametrize(
    "source_type, target_shape",
    [
        (
            DataShapeType.BATCH_CHANNEL_SEQUENCE,
            DataShapeType.BATCH_CHANNEL_SEQUENCE,
        ),
        (DataShapeType.BATCH_SEQUENCE, DataShapeType.BATCH_CHANNEL_SEQUENCE),
        (DataShapeType.BATCH_CHANNEL_SEQUENCE, DataShapeType.BATCH_SEQUENCE),
        (DataShapeType.BATCH_SEQUENCE, DataShapeType.BATCH_SEQUENCE),
    ],  # both source and target need to be batched (ecdf estimation required)
)
def test_calibrate(make_tensor, source_type, target_shape):
    source = make_tensor(source_type)
    target = make_tensor(target_shape)
    calibrated = calibrate(target, source)
    assert isinstance(calibrated, torch.Tensor)
    assert calibrated.shape == source.shape
