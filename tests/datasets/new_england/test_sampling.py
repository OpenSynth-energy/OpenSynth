import polars as pl

from opensynth.datasets.new_england import sampling


def _toy_metadata() -> pl.DataFrame:
    rows = []
    # NH: 20 buildings in cell (0, 1), 10 in (2, 0), 4 in (3, 2)
    for i in range(20):
        rows.append(("NH", 1000 + i, 0, 1))
    for i in range(10):
        rows.append(("NH", 2000 + i, 2, 0))
    for i in range(4):
        rows.append(("NH", 3000 + i, 3, 2))
    return pl.DataFrame(
        rows,
        schema=["state", "bldg_id", "archetype", "heating_fuel"],
        orient="row",
    )


def _toy_joint() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "state": ["NH", "NH", "NH"],
            "archetype": [0, 2, 3],
            "heating_fuel": [1, 0, 2],
            "share": [0.70, 0.25, 0.05],
        }
    )


class TestSelectBuildings:

    def test_total_and_proportions(self):
        manifest = sampling.select_buildings(
            _toy_metadata(),
            _toy_joint(),
            buildings_per_state=20,
            min_per_cell=2,
            seed=0,
        )
        assert len(manifest) == 20
        counts = {
            (r["archetype"], r["heating_fuel"]): r["n"]
            for r in manifest.group_by(["archetype", "heating_fuel"])
            .agg(pl.len().alias("n"))
            .iter_rows(named=True)
        }
        # Largest share gets the largest allocation
        assert counts[(0, 1)] > counts[(2, 0)] > 0

    def test_min_cell_floor(self):
        manifest = sampling.select_buildings(
            _toy_metadata(),
            _toy_joint(),
            buildings_per_state=20,
            min_per_cell=2,
            seed=0,
        )
        small_cell = manifest.filter(
            (pl.col("archetype") == 3) & (pl.col("heating_fuel") == 2)
        )
        # share 0.05 of 20 would floor to 1; min_per_cell lifts it
        assert len(small_cell) >= 2

    def test_seed_determinism(self):
        a = sampling.select_buildings(
            _toy_metadata(),
            _toy_joint(),
            buildings_per_state=20,
            min_per_cell=2,
            seed=7,
        )
        b = sampling.select_buildings(
            _toy_metadata(),
            _toy_joint(),
            buildings_per_state=20,
            min_per_cell=2,
            seed=7,
        )
        assert a.equals(b)

    def test_capped_by_availability(self):
        manifest = sampling.select_buildings(
            _toy_metadata(),
            _toy_joint(),
            buildings_per_state=100,
            min_per_cell=2,
            seed=0,
        )
        # Only 34 buildings exist in total
        assert len(manifest) == 34
        assert manifest["bldg_id"].n_unique() == 34
