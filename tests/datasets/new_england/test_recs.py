from pathlib import Path

import pytest

from opensynth.datasets.new_england import recs

FIXTURE = Path("tests/data/new_england/recs_toy.csv")


class TestLoadRecs:

    df = recs.load_recs(FIXTURE)

    def test_filters_to_new_england(self):
        assert len(self.df) == 7  # TX row dropped
        assert set(self.df["state"].unique()) == {"NH", "MA"}

    def test_archetype_encoding(self):
        by_state = self.df.sort(["state", "annual_kwh"])
        # MA: apt 5+ (TYPEHUQ 5) -> 2, SF attached (3) -> 1,
        # SF detached (2) -> 0
        assert by_state.filter(by_state["state"] == "MA")[
            "archetype"
        ].to_list() == [2, 1, 0]
        # NH row with TYPEHUQ 1 (mobile home) -> 3
        assert (
            3
            in self.df.filter(self.df["state"] == "NH")["archetype"].to_list()
        )

    def test_heating_fuel_encoding(self):
        # Wood (FUELHEAT 7) maps to other (4)
        wood = self.df.filter(self.df["annual_kwh"] == 6000)
        assert wood["heating_fuel"].to_list() == [4]
        # Fuel oil (3) maps to 2
        oil = self.df.filter(self.df["annual_kwh"] == 8000)
        assert oil["heating_fuel"].to_list() == [2]

    def test_der_flags_treat_not_applicable_as_no(self):
        na_row = self.df.filter(self.df["annual_kwh"] == 6000)
        assert na_row["has_pv"].to_list() == [0]
        assert na_row["has_ev"].to_list() == [0]


class TestDistributions:

    df = recs.load_recs(FIXTURE)

    def test_joint_distribution_sums_to_one_per_state(self):
        import polars as pl

        joint = recs.joint_distribution(self.df)
        sums = joint.group_by("state").agg(pl.col("share").sum().alias("s"))
        for s in sums["s"].to_list():
            assert s == pytest.approx(1.0)

    def test_joint_distribution_weighted(self):
        joint = recs.joint_distribution(self.df)
        # NH cell (archetype 0, fuel oil 2): weight 200 of 500 total
        nh_oil = joint.filter(
            (joint["state"] == "NH")
            & (joint["archetype"] == 0)
            & (joint["heating_fuel"] == 2)
        )
        assert nh_oil["share"].to_list()[0] == pytest.approx(0.4)

    def test_annual_kwh_summary_weighted_mean(self):
        summary = recs.annual_kwh_summary(self.df)
        nh = summary.filter(summary["state"] == "NH")
        # (100*8000 + 100*10000 + 200*5000 + 100*6000) / 500
        assert nh["mean_kwh"].to_list()[0] == pytest.approx(6800.0)

    def test_der_shares_weighted(self):
        shares = recs.der_shares(self.df)
        ma = shares.filter(shares["state"] == "MA")
        # PV: 100 of 500; EV: 300 of 500
        assert ma["pv_share"].to_list()[0] == pytest.approx(0.2)
        assert ma["ev_share"].to_list()[0] == pytest.approx(0.6)
