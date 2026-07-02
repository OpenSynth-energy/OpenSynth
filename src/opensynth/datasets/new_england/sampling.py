# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Stratified selection of EULP training buildings.

Allocates buildings per state across archetype x heating-fuel cells
proportionally to RECS NWEIGHT shares, with a minimum per occupied
cell, and samples building IDs deterministically. The resulting
manifest CSV is the reproducibility anchor for the training corpus.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from opensynth.datasets.new_england import config

logger = logging.getLogger(__name__)


def _allocate_cells(
    shares: dict[tuple[int, int], float],
    available: dict[tuple[int, int], int],
    n_total: int,
    min_per_cell: int,
) -> dict[tuple[int, int], int]:
    """
    Allocate n_total buildings across cells.

    Target counts follow RECS shares (restricted to cells that exist
    in the EULP metadata, renormalised), subject to a minimum per
    occupied cell and capped by availability. Any shortfall is
    redistributed to cells with spare capacity by largest share.

    Args:
        shares (dict): RECS share per (archetype, heating_fuel) cell.
        available (dict): Building count per cell in EULP metadata.
        n_total (int): Buildings to allocate for this state.
        min_per_cell (int): Minimum per occupied cell.

    Returns:
        dict: Allocation per cell.
    """
    cells = sorted(available)
    share_sum = sum(shares.get(c, 0.0) for c in cells)
    norm = {
        c: (shares.get(c, 0.0) / share_sum if share_sum > 0 else 0.0)
        for c in cells
    }

    alloc = {}
    for c in cells:
        target = int(np.floor(norm[c] * n_total))
        target = max(target, min_per_cell)
        alloc[c] = min(target, available[c])

    remainder = n_total - sum(alloc.values())

    if remainder > 0:
        # Distribute the shortfall by largest share, capped by capacity
        for c in sorted(cells, key=lambda c: norm[c], reverse=True):
            if remainder <= 0:
                break
            take = min(remainder, available[c] - alloc[c])
            alloc[c] += take
            remainder -= take
        if remainder > 0:
            logger.warning(
                f"⚠️ Only {n_total - remainder}/{n_total} buildings "
                "available across cells"
            )
    elif remainder < 0:
        # Minimum-per-cell floors overshot the target: trim from the
        # smallest-share cells, never below the minimum
        for c in sorted(cells, key=lambda c: norm[c]):
            if remainder == 0:
                break
            floor_c = min(min_per_cell, available[c])
            take = min(alloc[c] - floor_c, -remainder)
            alloc[c] -= take
            remainder += take

    return alloc


def select_buildings(
    df_metadata: pl.DataFrame,
    df_joint: pl.DataFrame,
    buildings_per_state: int = config.BUILDINGS_PER_STATE,
    min_per_cell: int = config.MIN_BUILDINGS_PER_CELL,
    seed: int = config.SAMPLING_SEED,
) -> pl.DataFrame:
    """
    Select a stratified, seeded training sample of buildings.

    Args:
        df_metadata (pl.DataFrame): EULP metadata from
            get_data.load_eulp_metadata (bldg_id, state, archetype,
            heating_fuel).
        df_joint (pl.DataFrame): RECS joint distribution from
            recs.joint_distribution (state, archetype, heating_fuel,
            share).
        buildings_per_state (int): Sample size per state.
        min_per_cell (int): Minimum buildings per occupied cell.
        seed (int): RNG seed; same inputs and seed give the same
            manifest.

    Returns:
        pl.DataFrame: Manifest with columns state, bldg_id,
        archetype, heating_fuel.
    """
    rng = np.random.default_rng(seed)
    selected = []

    for state in config.NE_STATES:
        md = df_metadata.filter(pl.col("state") == state)
        if md.is_empty():
            continue
        shares = {
            (r["archetype"], r["heating_fuel"]): r["share"]
            for r in df_joint.filter(pl.col("state") == state).iter_rows(
                named=True
            )
        }
        available_df = md.group_by(["archetype", "heating_fuel"]).agg(
            pl.len().alias("n")
        )
        available = {
            (r["archetype"], r["heating_fuel"]): r["n"]
            for r in available_df.iter_rows(named=True)
        }

        alloc = _allocate_cells(
            shares, available, buildings_per_state, min_per_cell
        )

        for (archetype, fuel), n_cell in sorted(alloc.items()):
            if n_cell == 0:
                continue
            ids = (
                md.filter(
                    (pl.col("archetype") == archetype)
                    & (pl.col("heating_fuel") == fuel)
                )["bldg_id"]
                .sort()
                .to_numpy()
            )
            chosen = rng.choice(ids, size=n_cell, replace=False)
            selected.append(
                pl.DataFrame(
                    {
                        "state": [state] * n_cell,
                        "bldg_id": sorted(int(i) for i in chosen),
                        "archetype": [archetype] * n_cell,
                        "heating_fuel": [fuel] * n_cell,
                    }
                )
            )
        logger.info(
            f"🎯 {state}: {sum(alloc.values())} buildings across "
            f"{len([n for n in alloc.values() if n > 0])} cells"
        )

    return pl.concat(selected).sort(["state", "archetype", "heating_fuel"])


def write_manifest(manifest: pl.DataFrame, data_dir: str = "./data") -> Path:
    """
    Write the building manifest CSV.

    Args:
        manifest (pl.DataFrame): Output of select_buildings.
        data_dir (str): Data directory. Defaults to "./data".

    Returns:
        Path: Path of the written manifest.
    """
    out = Path(data_dir) / "raw/new_england/building_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(out)
    logger.info(f"📦 Wrote building manifest: {out}")
    return out
