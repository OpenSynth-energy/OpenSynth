# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Generate the New England synthetic dataset.

Per home: (state, archetype, heating_fuel) drawn from the RECS 2020
within-state joint distribution with states weighted by RECS household
weights; DER flags from RECS state adoption shares; then one 96-interval
day sampled per 2018 calendar date by conditioning the GMM on the
home's labels and its state's real 2018 daily temp-bin trajectory.

Magnitude calibration: day-independent conditional sampling collapses
per-home annual variance (each home's 365 draws average toward its
segment mean, losing persistent home-level identity). To restore the
cross-home annual-kWh distribution, every home receives one persistent
scale factor drawn from the empirical distribution of relative annuals
(home annual / segment mean) among real training-split homes of the
same (archetype, heating_fuel) segment. Shapes, timing and DER
signatures are unaffected; the calibration is disclosed in the
dataset card.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch

from opensynth.datasets.new_england import config, recs, weather
from opensynth.datasets.new_england.faraday_ne import NewEnglandFaradayModel
from opensynth.models.faraday.gaussian_mixture import GaussianMixtureModel
from opensynth.models.faraday.vae_model import FaradayVAE

logger = logging.getLogger(__name__)

# Clamp for the per-home magnitude scale: covers the observed range of
# real relative annuals while preventing pathological PV-system
# scaling in the tails.
SCALE_CLAMP = (0.25, 4.0)
MIN_SEGMENT_HOMES = 20


def load_model(
    model_dir: Path, gmm_k: int = 200
) -> tuple[NewEnglandFaradayModel, dict]:
    """
    Rebuild a NewEnglandFaradayModel from training checkpoints.

    Args:
        model_dir (Path): Directory with ne_vae.ckpt,
            ne_vae_meta.json and ne_gmm_{k}.pt.
        gmm_k (int): GMM component count to load.

    Returns:
        tuple: (model, vae meta dict).
    """
    vae = FaradayVAE.load_from_checkpoint(
        model_dir / "ne_vae.ckpt", map_location="cpu"
    )
    meta = json.loads((model_dir / "ne_vae_meta.json").read_text())
    vae.feature_list = meta["feature_list"]
    vae.eval()

    ckpt = torch.load(model_dir / f"ne_gmm_{gmm_k}.pt", map_location="cpu")
    gmm = GaussianMixtureModel(
        num_components=gmm_k,
        num_features=vae.latent_dim + len(meta["feature_list"]),
        reg_covar=ckpt["covariance_reg"],
    )
    state = dict(ckpt["gmm_state_dict"])
    # Backward compatibility: checkpoints written before
    # GaussianMixtureModel.update_params normalised the nll shape
    # carry a 0-dim buffer
    state["nll"] = state["nll"].reshape(1)
    gmm.load_state_dict(state)

    model = NewEnglandFaradayModel(
        vae_module=vae,
        n_components=gmm_k,
        covariance_reg=ckpt["covariance_reg"],
    )
    model.gmm_module = gmm
    model.feature_range = ckpt["feature_range"]
    return model, meta


def draw_homes(
    df_recs: pl.DataFrame, n_homes: int, rng: np.random.Generator
) -> pl.DataFrame:
    """
    Draw per-home labels from the RECS 2020 joint distribution.

    Args:
        df_recs (pl.DataFrame): Output of recs.load_recs.
        n_homes (int): Number of homes.
        rng (np.random.Generator): Seeded generator.

    Returns:
        pl.DataFrame: home_id, state_postal, state, archetype,
        heating_fuel, has_pv, has_ev.
    """
    joint = recs.joint_distribution(df_recs)
    ders = recs.der_shares(df_recs)
    state_w = (
        df_recs.group_by("state")
        .agg(pl.col("nweight").sum().alias("w"))
        .sort("state")
    )
    state_probs = (state_w["w"] / state_w["w"].sum()).to_numpy()
    states = state_w["state"].to_list()
    der_lookup = {
        r["state"]: (r["pv_share"], r["ev_share"])
        for r in ders.iter_rows(named=True)
    }

    rows = []
    for i in range(n_homes):
        state = states[rng.choice(len(states), p=state_probs)]
        cells = joint.filter(pl.col("state") == state)
        shares = cells["share"].to_numpy()
        cell = cells.row(
            int(rng.choice(len(cells), p=shares / shares.sum())), named=True
        )
        pv_share, ev_share = der_lookup[state]
        rows.append(
            {
                "home_id": f"NE_{i:04d}",
                "state_postal": state,
                "state": config.STATE_ENCODING[state],
                "archetype": cell["archetype"],
                "heating_fuel": cell["heating_fuel"],
                "has_pv": int(rng.random() < pv_share),
                "has_ev": int(rng.random() < ev_share),
            }
        )
    return pl.DataFrame(rows)


def relative_annual_table(train_data_path: Path) -> pl.DataFrame:
    """
    Relative annual consumption of real training homes per segment.

    Args:
        train_data_path (Path): Packed train data.csv.

    Returns:
        pl.DataFrame: Columns archetype, heating_fuel, rel_annual
        (home annual / segment mean annual), one row per home.
    """
    per_home = (
        pl.scan_csv(train_data_path)
        .select(
            "ID",
            "archetype",
            "heating_fuel",
            pl.col("kwh").str.json_decode(pl.List(pl.Float64)).list.sum(),
        )
        .group_by("ID")
        .agg(
            pl.col("archetype").first(),
            pl.col("heating_fuel").first(),
            pl.col("kwh").sum().alias("annual"),
        )
        .collect()
    )
    return (
        per_home.with_columns(
            (
                pl.col("annual")
                / pl.col("annual").mean().over(["archetype", "heating_fuel"])
            ).alias("rel_annual")
        )
        .select("archetype", "heating_fuel", "rel_annual")
        .sort(["archetype", "heating_fuel"])
    )


def draw_scales(
    homes: pl.DataFrame,
    rel_table: pl.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one persistent magnitude scale per home.

    Samples from the empirical relative-annual distribution of the
    home's (archetype, heating_fuel) segment, falling back to the
    pooled distribution for segments with fewer than
    MIN_SEGMENT_HOMES real homes. Clamped to SCALE_CLAMP.

    Args:
        homes (pl.DataFrame): Output of draw_homes.
        rel_table (pl.DataFrame): Output of relative_annual_table.
        rng (np.random.Generator): Seeded generator.

    Returns:
        np.ndarray: Scale factor per home [n_homes].
    """
    pooled = rel_table["rel_annual"].to_numpy()
    scales = np.empty(homes.height)
    for i, row in enumerate(homes.iter_rows(named=True)):
        seg = rel_table.filter(
            (pl.col("archetype") == row["archetype"])
            & (pl.col("heating_fuel") == row["heating_fuel"])
        )["rel_annual"].to_numpy()
        source = seg if len(seg) >= MIN_SEGMENT_HOMES else pooled
        scales[i] = source[rng.integers(len(source))]
    return scales.clip(*SCALE_CLAMP)


def build_calendar(
    homes: pl.DataFrame, temp_table: pl.DataFrame
) -> pd.DataFrame:
    """
    One row per home-day with all conditioning labels.

    Args:
        homes (pl.DataFrame): Output of draw_homes.
        temp_table (pl.DataFrame): weather.build_temp_bin_table output.

    Returns:
        pd.DataFrame: home-day rows with config.FEATURE_COLS labels,
        home_id and date, sorted by home then date.
    """
    tt = temp_table.rename({"state": "state_postal"}).to_pandas()
    tt["date"] = pd.to_datetime(tt["date"])
    calendar = homes.to_pandas().merge(
        tt[["state_postal", "date", "temp_bin"]], on="state_postal"
    )
    calendar["month"] = calendar["date"].dt.month
    calendar["dayofweek"] = calendar["date"].dt.dayofweek
    return calendar.sort_values(["home_id", "date"]).reset_index(drop=True)


def generate_profiles(
    model: NewEnglandFaradayModel,
    calendar: pd.DataFrame,
    kwh_mean: float,
    kwh_std: float,
) -> np.ndarray:
    """
    One synthetic day per calendar row, batched by label combination.

    Net load stays negative only for has_pv homes.

    Args:
        model (NewEnglandFaradayModel): Loaded model.
        calendar (pd.DataFrame): Output of build_calendar.
        kwh_mean (float): Training standardisation mean.
        kwh_std (float): Training standardisation stdev.

    Returns:
        np.ndarray: kWh per 15-min interval [n_rows, 96].
    """
    feats = list(config.FEATURE_COLS)
    out = np.empty((len(calendar), 96), dtype=np.float32)
    groups = calendar.groupby(feats).indices
    logger.info(
        f"🎨 Generating {len(calendar)} home-days "
        f"({len(groups)} label combinations)"
    )
    for n, (key, idx) in enumerate(groups.items()):
        labels = dict(zip(feats, (float(v) for v in key)))
        with torch.no_grad():
            sampled = model.sample_gmm_conditional(labels, len(idx))
        kwh = sampled["kwh"].numpy() * kwh_std + kwh_mean
        if labels["has_pv"] == 0:
            kwh = kwh.clip(min=0)
        out[idx] = kwh
        if (n + 1) % 5000 == 0:
            logger.info(f"  {n + 1}/{len(groups)} combinations")
    return out


def write_dataset(
    out_dir: Path,
    homes: pl.DataFrame,
    calendar: pd.DataFrame,
    kwh: np.ndarray,
    summary: dict,
    write_csv: bool = True,
) -> None:
    """
    Write parquet, csv.gz, metadata and summary files.

    Args:
        out_dir (Path): Output directory.
        homes (pl.DataFrame): Per-home metadata incl. magnitude_scale.
        calendar (pd.DataFrame): Home-day rows aligned with kwh.
        kwh (np.ndarray): [n_rows, 96] kWh values.
        summary (dict): Run summary for generation_summary.json.
        write_csv (bool): Also write the csv.gz copy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name files by their actual home count so smaller test runs
    # neither masquerade as nor clobber the release artifact
    stem = f"ne_synthetic_{homes.height}homes"

    offsets = (np.arange(96) * np.timedelta64(15, "m")).astype(
        "timedelta64[ns]"
    )
    timestamps = (
        calendar["date"].to_numpy()[:, None] + offsets[None, :]
    ).ravel()
    df_long = pl.DataFrame(
        {
            "home_id": np.repeat(calendar["home_id"].to_numpy(), 96),
            "timestamp": timestamps,
            "kwh": kwh.ravel(),
        }
    ).with_columns(pl.col("home_id").cast(pl.Categorical))

    parquet_path = out_dir / f"{stem}.parquet"
    df_long.write_parquet(parquet_path)
    logger.info(f"💾 Wrote {df_long.height:,} rows to {parquet_path}")

    if write_csv:
        csv_path = out_dir / f"{stem}.csv.gz"
        with gzip.open(csv_path, "wb", compresslevel=6) as f:
            df_long.write_csv(f)
        logger.info(f"💾 Wrote {csv_path}")

    homes.write_csv(out_dir / f"{stem}_metadata.csv")
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info(f"💾 Wrote metadata and summary to {out_dir}")


def generate_ne_dataset(
    data_dir: str = "./data",
    n_homes: int = 1000,
    gmm_k: int = 200,
    seed: int = config.SAMPLING_SEED,
    out_dir: Optional[str] = None,
    calibrate: bool = True,
    write_csv: bool = True,
) -> None:
    """
    Full dataset generation: draw homes, generate, calibrate, write.

    Args:
        data_dir (str): Data directory.
        n_homes (int): Homes to generate.
        gmm_k (int): GMM checkpoint to use.
        seed (int): RNG seed.
        out_dir (str, optional): Output directory. Defaults to
            data/synthetic/new_england.
        calibrate (bool): Apply per-home magnitude calibration.
        write_csv (bool): Also write the csv.gz copy.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data_path = Path(data_dir)
    out_path = (
        Path(out_dir) if out_dir else data_path / "synthetic/new_england"
    )
    model_dir = data_path / "models/new_england"

    model, meta = load_model(model_dir, gmm_k)
    stats = pl.read_csv(data_path / "processed/new_england/train/mean_std.csv")
    kwh_mean, kwh_std = stats["mean"][0], stats["stdev"][0]

    df_recs = recs.load_recs(data_path / config.RECS_CSV_RELPATH)
    homes = draw_homes(df_recs, n_homes, rng)
    logger.info(
        f"🏠 Drew {n_homes} homes: "
        f"{homes['has_pv'].sum()} PV, {homes['has_ev'].sum()} EV"
    )

    if calibrate:
        rel_table = relative_annual_table(
            data_path / "processed/new_england/train/data.csv"
        )
        scales = draw_scales(homes, rel_table, rng)
        logger.info(
            f"⚖️ Magnitude calibration: scale mean {scales.mean():.3f}, "
            f"range [{scales.min():.3f}, {scales.max():.3f}]"
        )
    else:
        scales = np.ones(homes.height)
    homes = homes.with_columns(pl.Series("magnitude_scale", scales.round(4)))

    temp_table = weather.build_temp_bin_table(str(data_path))
    calendar = build_calendar(homes, temp_table)
    kwh = generate_profiles(model, calendar, kwh_mean, kwh_std)

    scale_per_row = (
        calendar[["home_id"]]
        .merge(
            homes.to_pandas()[["home_id", "magnitude_scale"]], on="home_id"
        )["magnitude_scale"]
        .to_numpy()
    )
    kwh = kwh * scale_per_row[:, None].astype(np.float32)

    annual = (
        pd.Series(kwh.sum(axis=1))
        .groupby(calendar["home_id"])
        .sum()
        .to_numpy()
    )
    recs_mean = float(
        (df_recs["annual_kwh"] * df_recs["nweight"]).sum()
        / df_recs["nweight"].sum()
    )
    summary = {
        "n_homes": n_homes,
        "seed": seed,
        "gmm_k": gmm_k,
        "calibrated": calibrate,
        "vae_final_loss": meta["epoch_losses"][-1],
        "annual_kwh_mean": float(annual.mean()),
        "annual_kwh_std": float(annual.std()),
        "annual_kwh_deciles": {
            f"q{int(q * 100)}": float(np.quantile(annual, q))
            for q in np.arange(0.1, 1.0, 0.1)
        },
        "recs_weighted_mean_kwh": recs_mean,
        "mpe_vs_recs_pct": float(
            (annual.mean() - recs_mean) / recs_mean * 100
        ),
        "pv_homes": int(homes["has_pv"].sum()),
        "ev_homes": int(homes["has_ev"].sum()),
    }
    logger.info(
        f"📊 Annual kWh: mean {summary['annual_kwh_mean']:,.0f} "
        f"(RECS {recs_mean:,.0f}, MPE {summary['mpe_vs_recs_pct']:+.1f}%), "
        f"std {summary['annual_kwh_std']:,.0f}"
    )
    write_dataset(out_path, homes, calendar, kwh, summary, write_csv)
    logger.info("🎉 Dataset generation complete")
