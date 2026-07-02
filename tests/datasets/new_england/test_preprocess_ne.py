from datetime import datetime, timedelta

import polars as pl
import pytest

from opensynth.datasets.new_england import preprocess_ne


def _toy_eulp_parquet(tmp_path, n_days: int = 3):
    """EULP-like parquet: period-ending timestamps, 00:15 start."""
    start = datetime(2018, 1, 1, 0, 15)
    n = n_days * 96
    df = pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=15 * i) for i in range(n)],
            preprocess_ne.EULP_KWH_COL: [0.1] * n,
        }
    )
    path = tmp_path / "NH_1-0.parquet"
    df.write_parquet(path)
    return path


class TestMeltBuilding:

    def test_timestamps_shift_to_period_beginning(self, tmp_path):
        df = preprocess_ne.melt_building(_toy_eulp_parquet(tmp_path), "NH_1")
        first = df["DateTime"].min()
        assert first == datetime(2018, 1, 1, 0, 0)

    def test_day_boundary_owns_96_intervals(self, tmp_path):
        # The timestamp-convention test: after the shift, every
        # calendar day must own exactly 96 readings. Without the
        # shift, day 1 would have 95 and the last reading would
        # bleed into the following day. A one-hour class of error
        # here would silently consume the entire peak-timing
        # validation tolerance.
        df = preprocess_ne.melt_building(
            _toy_eulp_parquet(tmp_path, n_days=3), "NH_1"
        )
        per_day = (
            df.with_columns(pl.col("DateTime").dt.date().alias("date"))
            .group_by("date")
            .len()
            .sort("date")
        )
        assert per_day["len"].to_list() == [96, 96, 96]

    def test_schema_and_id(self, tmp_path):
        df = preprocess_ne.melt_building(_toy_eulp_parquet(tmp_path), "NH_1")
        assert df.columns == ["ID", "DateTime", "kwh"]
        assert df["ID"].unique().to_list() == ["NH_1"]
        assert df["kwh"].sum() == pytest.approx(3 * 96 * 0.1)


class TestPackChunk:

    def test_packs_96_interval_days_with_ne_features(self, tmp_path):
        df = preprocess_ne.melt_building(
            _toy_eulp_parquet(tmp_path, n_days=2), "NH_1"
        )
        df = df.with_columns(
            pl.lit(3).alias("state"),
            pl.lit(0).alias("archetype"),
            pl.lit(2).alias("heating_fuel"),
            pl.lit(0).alias("has_ev"),
            pl.lit(0).alias("has_pv"),
            pl.lit(1).alias("temp_bin"),
        )
        packed = preprocess_ne._pack_chunk(df)
        assert len(packed) == 2
        assert all(len(k) == 96 for k in packed["kwh"])
        for col in preprocess_ne.NE_FEATURE_COLS:
            assert col in packed.columns
        # Integer conditioning labels survive the packing groupby
        assert packed["state"].tolist() == [3, 3]
        assert packed["temp_bin"].tolist() == [1, 1]
