from opensynth.datasets.new_england import get_data


class TestDownloadIfMissing:

    def test_skips_existing_non_parquet(self, tmp_path, monkeypatch):
        target = tmp_path / "data.csv"
        target.write_bytes(b"some,data")
        called = []
        monkeypatch.setattr(
            get_data.datasets_utils,
            "download_data",
            lambda url, path: called.append(url),
        )
        assert get_data._download_if_missing("http://x", target) is False
        assert called == []

    def test_skips_intact_parquet(self, tmp_path, monkeypatch):
        target = tmp_path / "data.parquet"
        target.write_bytes(b"header-bytes" + b"PAR1")
        called = []
        monkeypatch.setattr(
            get_data.datasets_utils,
            "download_data",
            lambda url, path: called.append(url),
        )
        assert get_data._download_if_missing("http://x", target) is False
        assert called == []

    def test_redownloads_truncated_parquet(self, tmp_path, monkeypatch):
        # A download killed mid-write leaves a file without the
        # parquet magic footer; the resume loop must not trust it
        target = tmp_path / "data.parquet"
        target.write_bytes(b"truncated-partial-content")

        def fake_download(url, path):
            path.write_bytes(b"fresh" + b"PAR1")

        monkeypatch.setattr(
            get_data.datasets_utils, "download_data", fake_download
        )
        assert get_data._download_if_missing("http://x", target) is True
        assert target.read_bytes().endswith(b"PAR1")

    def test_downloads_missing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "new.parquet"

        def fake_download(url, path):
            path.write_bytes(b"fresh" + b"PAR1")

        monkeypatch.setattr(
            get_data.datasets_utils, "download_data", fake_download
        )
        assert get_data._download_if_missing("http://x", target) is True
        assert target.exists()
