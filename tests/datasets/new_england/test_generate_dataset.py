# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import polars as pl

from opensynth.datasets.new_england import generate_dataset, recs

RECS_FIXTURE = Path("tests/data/new_england/recs_toy.csv")


def _toy_recs() -> pl.DataFrame:
    return recs.load_recs(RECS_FIXTURE)


class TestDrawHomes:

    homes = generate_dataset.draw_homes(
        _toy_recs(), n_homes=50, rng=np.random.default_rng(0)
    )

    def test_shape_and_columns(self):
        assert self.homes.height == 50
        assert set(self.homes.columns) == {
            "home_id",
            "state_postal",
            "state",
            "archetype",
            "heating_fuel",
            "has_pv",
            "has_ev",
        }

    def test_states_come_from_recs(self):
        # The toy fixture only has NH and MA households
        assert set(self.homes["state_postal"].unique()) <= {"NH", "MA"}

    def test_state_encoding_consistent(self):
        from opensynth.datasets.new_england import config

        for row in self.homes.iter_rows(named=True):
            assert row["state"] == config.STATE_ENCODING[row["state_postal"]]

    def test_deterministic_for_seed(self):
        again = generate_dataset.draw_homes(
            _toy_recs(), n_homes=50, rng=np.random.default_rng(0)
        )
        assert again.equals(self.homes)


class TestDrawScales:

    def _rel_table(self, n_per_segment: int) -> pl.DataFrame:
        rng = np.random.default_rng(1)
        return pl.DataFrame(
            {
                "archetype": [0] * n_per_segment,
                "heating_fuel": [1] * n_per_segment,
                "rel_annual": rng.uniform(0.5, 1.5, n_per_segment),
            }
        )

    def _homes(self, archetype: int, heating_fuel: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "home_id": ["NE_0000"],
                "state_postal": ["NH"],
                "state": [3],
                "archetype": [archetype],
                "heating_fuel": [heating_fuel],
                "has_pv": [0],
                "has_ev": [0],
            }
        )

    def test_draws_from_segment_when_populated(self):
        table = self._rel_table(generate_dataset.MIN_SEGMENT_HOMES)
        scales = generate_dataset.draw_scales(
            self._homes(0, 1), table, np.random.default_rng(0)
        )
        assert scales[0] in table["rel_annual"].to_numpy()

    def test_falls_back_to_pooled_for_sparse_segment(self):
        table = self._rel_table(generate_dataset.MIN_SEGMENT_HOMES)
        # Home in a segment absent from the table: pooled fallback
        scales = generate_dataset.draw_scales(
            self._homes(3, 4), table, np.random.default_rng(0)
        )
        assert scales[0] in table["rel_annual"].to_numpy()

    def test_clamped(self):
        table = pl.DataFrame(
            {
                "archetype": [0] * 25,
                "heating_fuel": [1] * 25,
                "rel_annual": [100.0] * 25,
            }
        )
        scales = generate_dataset.draw_scales(
            self._homes(0, 1), table, np.random.default_rng(0)
        )
        assert scales[0] == generate_dataset.SCALE_CLAMP[1]


class TestRelativeAnnualTable:

    def test_relative_annuals_mean_one_per_segment(self, tmp_path):
        # Two homes in one segment, 2:1 annual ratio
        rows = []
        for home, kwh in [("A", "[1.0, 1.0]"), ("B", "[2.0, 2.0]")]:
            for day in ["2018-01-01", "2018-01-02"]:
                rows.append(
                    {
                        "ID": home,
                        "archetype": 0,
                        "heating_fuel": 1,
                        "kwh": kwh,
                        "date": day,
                    }
                )
        path = tmp_path / "data.csv"
        pl.DataFrame(rows).write_csv(path)

        table = generate_dataset.relative_annual_table(path)
        rel = sorted(table["rel_annual"].to_list())
        assert np.isclose(rel[0], 2 / 3)
        assert np.isclose(rel[1], 4 / 3)


class TestBuildCalendar:

    def test_one_row_per_home_day(self):
        homes = pl.DataFrame(
            {
                "home_id": ["NE_0000", "NE_0001"],
                "state_postal": ["NH", "MA"],
                "state": [3, 1],
                "archetype": [0, 1],
                "heating_fuel": [1, 2],
                "has_pv": [0, 1],
                "has_ev": [0, 0],
            }
        )
        temp_table = pl.DataFrame(
            {
                "state": ["NH", "NH", "MA", "MA"],
                "date": ["2018-01-01", "2018-01-02"] * 2,
                "tmean_c": [-5.0, -2.0, -1.0, 3.0],
                "temp_bin": [2, 3, 3, 3],
            }
        )
        calendar = generate_dataset.build_calendar(homes, temp_table)
        assert len(calendar) == 4
        assert list(calendar.columns[-2:]) == ["month", "dayofweek"]
        nh = calendar[calendar["home_id"] == "NE_0000"]
        assert nh["temp_bin"].tolist() == [2, 3]
        # 2018-01-01 was a Monday
        assert nh["dayofweek"].tolist() == [0, 1]
