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


class TestLoadIsoneHourly:

    def test_smd_export_with_hour_ending_column(self, tmp_path):
        csv = tmp_path / "smd.csv"
        csv.write_text(
            "Date,Hr_End,DEMAND\n"
            "2019-01-15,1,10000\n"
            "2019-01-15,2,11000\n"
            "2019-01-15,19,15000\n"
        )
        df = get_data.load_isone_hourly(csv)
        hours = [t.hour for t in df["timestamp"].to_list()]
        # Hour-ending 1 covers 00:00-01:00 -> period-beginning 0
        assert hours == [0, 1, 18]

    def test_eia_export_with_utc_offset(self, tmp_path):
        csv = tmp_path / "eia.csv"
        csv.write_text(
            "period,Demand (MWh)\n"
            "2019-01-15T23,15000\n"
            "2019-01-16T00,14000\n"
        )
        df = get_data.load_isone_hourly(csv, utc_offset_hours=-5)
        hours = [t.hour for t in df["timestamp"].to_list()]
        assert hours == [18, 19]


class TestDownloadRetryCoverage:

    def test_incomplete_read_is_retried(self, tmp_path, monkeypatch):
        import http.client

        calls = []

        def flaky(url, out_path):
            calls.append(url)
            if len(calls) == 1:
                out_path.write_bytes(b"partial")
                raise http.client.IncompleteRead(b"partial")
            out_path.write_bytes(b"complete")

        monkeypatch.setattr(get_data.datasets_utils, "download_data", flaky)
        monkeypatch.setattr(get_data.time, "sleep", lambda s: None)
        out = tmp_path / "file.csv"
        assert get_data._download_if_missing("http://x", out) is True
        assert len(calls) == 2
        assert out.read_bytes() == b"complete"
