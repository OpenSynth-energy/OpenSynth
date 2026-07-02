from datetime import datetime, timedelta

import polars as pl
import pytest

from opensynth.datasets.new_england import config, der_augmentation


def _toy_long(has_ev: int = 1, has_pv: int = 0, temp_bin: int = 5):
    """Two homes x one day x 96 fifteen-minute rows each."""
    start = datetime(2018, 6, 5, 0, 0)
    rows = []
    for home, ev, pv in (("H1", has_ev, has_pv), ("H2", 0, 0)):
        for i in range(96):
            rows.append(
                (
                    home,
                    start + timedelta(minutes=15 * i),
                    0.1,
                    "NH",
                    temp_bin,
                    ev,
                    pv,
                )
            )
    return pl.DataFrame(
        rows,
        schema=[
            "ID",
            "DateTime",
            "kwh",
            "state",
            "temp_bin",
            "has_ev",
            "has_pv",
        ],
        orient="row",
    )


def _flat_shape(kw_per_kw: float = 0.5) -> pl.DataFrame:
    """Toy PV shape: constant output 10:00-14:00 in June."""
    return pl.DataFrame(
        {
            "site": ["north"] * 4,
            "month": [6] * 4,
            "hour": [10, 11, 12, 13],
            "kw_per_kw": [kw_per_kw] * 4,
        }
    )


class TestInjectEv:

    def test_non_flagged_homes_untouched(self):
        df = der_augmentation.inject_ev(_toy_long(), seed=0)
        h2 = df.filter(pl.col("ID") == "H2")
        assert h2["kwh"].sum() == pytest.approx(96 * 0.1)

    def test_ev_energy_added_in_evening_window(self):
        df = der_augmentation.inject_ev(_toy_long(), seed=0)
        h1 = df.filter(pl.col("ID") == "H1").sort("DateTime")
        added = h1["kwh"].sum() - 96 * 0.1
        # Charged (p=0.75 under this seed): energy within the
        # distribution's plausible range
        assert added > 0.5
        assert added < (
            config.EV_ENERGY_MEAN_KWH + 4 * config.EV_ENERGY_STD_KWH
        )
        # Nothing added before the arrival window opens (16:00)
        before_window = h1[: 16 * 4]["kwh"].sum()
        assert before_window == pytest.approx(16 * 4 * 0.1)

    def test_winter_energy_factor(self):
        mild = der_augmentation.inject_ev(_toy_long(temp_bin=5), seed=3)
        cold = der_augmentation.inject_ev(_toy_long(temp_bin=1), seed=3)
        added_mild = mild.filter(pl.col("ID") == "H1")["kwh"].sum()
        added_cold = cold.filter(pl.col("ID") == "H1")["kwh"].sum()
        # Same seed, same draws; cold day scales energy by 1.3
        # (unless truncated at midnight, which seed 3 avoids)
        assert added_cold > added_mild

    def test_seed_determinism(self):
        a = der_augmentation.inject_ev(_toy_long(), seed=11)
        b = der_augmentation.inject_ev(_toy_long(), seed=11)
        assert a.sort(["ID", "DateTime"]).equals(b.sort(["ID", "DateTime"]))


class TestApplyPv:

    def test_pv_subtracts_and_can_go_negative(self):
        df = der_augmentation.apply_pv(
            _toy_long(has_ev=0, has_pv=1), _flat_shape(0.5), seed=0
        )
        h1 = df.filter(pl.col("ID") == "H1").sort("DateTime")
        # Any system size (>= 4 kW) at 0.5 kW/kW for 15 min far
        # exceeds the 0.1 kWh baseline load
        midday = h1[10 * 4 : 14 * 4]["kwh"]
        assert (midday < 0).all()
        # Outside the shape's hours, load is unchanged
        assert h1[0]["kwh"].item() == pytest.approx(0.1)

    def test_non_flagged_homes_untouched(self):
        df = der_augmentation.apply_pv(
            _toy_long(has_ev=0, has_pv=1), _flat_shape(0.5), seed=0
        )
        h2 = df.filter(pl.col("ID") == "H2")
        assert h2["kwh"].sum() == pytest.approx(96 * 0.1)


class TestAssignDerFlags:

    def _homes(self, n_per_archetype: int = 500) -> pl.DataFrame:
        rows = []
        for archetype in (0, 2):
            for i in range(n_per_archetype):
                rows.append((f"A{archetype}_{i}", "NH", archetype))
        return pl.DataFrame(
            rows, schema=["ID", "state", "archetype"], orient="row"
        )

    def _shares(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"state": ["NH"], "pv_share": [0.2], "ev_share": [0.3]}
        )

    def test_expected_shares_preserved(self):
        df = der_augmentation.assign_der_flags(
            self._homes(), self._shares(), seed=0
        )
        assert df["has_ev"].mean() == pytest.approx(0.3, abs=0.05)
        assert df["has_pv"].mean() == pytest.approx(0.2, abs=0.05)

    def test_pv_weighted_to_single_family(self):
        df = der_augmentation.assign_der_flags(
            self._homes(), self._shares(), seed=0
        )
        sf_rate = df.filter(pl.col("archetype") == 0)["has_pv"].mean()
        mf_rate = df.filter(pl.col("archetype") == 2)["has_pv"].mean()
        assert sf_rate > mf_rate

    def test_seed_determinism(self):
        a = der_augmentation.assign_der_flags(
            self._homes(), self._shares(), seed=5
        )
        b = der_augmentation.assign_der_flags(
            self._homes(), self._shares(), seed=5
        )
        assert a.equals(b)
