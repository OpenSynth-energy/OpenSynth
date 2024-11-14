from pathlib import Path

import litdata as ld
import pytest
import torch

from opensynth.data_modules.streaming_data_module import (
    StreamData,
    StreamDataModule,
)

TEST_STREAM_PATH = Path("tests/data/streaming")


@pytest.fixture()
def dummy_dataset():
    return StreamData(
        data_path=str(TEST_STREAM_PATH),
        stats_path=TEST_STREAM_PATH / "mean_std.csv",
    )


@pytest.fixture()
def dummy_data_module():
    return StreamDataModule(
        data_path=str(TEST_STREAM_PATH),
        stats_path=TEST_STREAM_PATH / "mean_std.csv",
    )


class TestStreamDataset:
    @pytest.fixture(autouse=True)
    def _stream_dataset(self, dummy_dataset):
        self.stream_dataset = dummy_dataset

    def test_setup(self):
        assert self.stream_dataset.feature_mean == 1.0
        assert self.stream_dataset.feature_std == 2.0

    def test_standardise(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        got = self.stream_dataset.standardise(x)
        expected = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(got, expected, atol=1e-4)

    def test_reconstruct(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        got = self.stream_dataset.reconstruct(x)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(got, expected, atol=1e-4)

    def test_getitem(self):
        idx = 0
        got = self.stream_dataset[idx]
        assert isinstance(got, dict)
        assert set(got.keys()) == {"kwh", "features"}

    def test_len(self):
        assert len(self.stream_dataset) == 100


class TestDataModule:
    @pytest.fixture(autouse=True)
    def _stream_data_module(self, dummy_data_module, monkeypatch):
        monkeypatch.setenv("CLOUD_ML_PROJECT_ID", "test_proj")
        self.data_module = dummy_data_module
        self.data_module.setup(stage="")

    def test_setup(self):
        assert isinstance(self.data_module.train_dataset, StreamData)

    def test_train_dataloader(self):
        dl = self.data_module.train_dataloader()
        assert isinstance(dl, ld.StreamingDataLoader)

    def test_reconstruct_kwh(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        got = self.data_module.reconstruct_kwh(x)
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(got, expected, atol=1e-4)
