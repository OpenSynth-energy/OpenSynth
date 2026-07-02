# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
import io
import urllib.request

import pytest

from opensynth.datasets import datasets_utils


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadDataAtomicity:

    def test_successful_download_lands_at_target(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _FakeResponse(b"payload"),
        )
        target = tmp_path / "file.csv"
        datasets_utils.download_data("http://x", target)
        assert target.read_bytes() == b"payload"
        assert not target.with_suffix(".csv.part").exists()

    def test_failed_download_leaves_no_file(self, tmp_path, monkeypatch):
        class _Exploding(_FakeResponse):
            def read(self, n=-1):
                raise ConnectionResetError("mid-body close")

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda request: _Exploding(b""),
        )
        target = tmp_path / "file.csv"
        with pytest.raises(OSError):
            datasets_utils.download_data("http://x", target)
        # The interrupted transfer must not leave a truncated file at
        # the target path for skip-if-exists logic to trust
        assert not target.exists()
        assert not target.with_suffix(".csv.part").exists()
