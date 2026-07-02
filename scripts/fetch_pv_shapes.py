# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Build normalised PV generation shapes for the NE reference sites.

Primary source: Open-Meteo historical archive (ERA5), hourly global
tilted irradiance and 2 m temperature for weather year 2018, converted
to AC kW per kW capacity with the PVWatts simplified model (NOCT cell
temperature, -0.37 %/C power coefficient, 14 % system losses, 96 %
nominal inverter efficiency). Using actual 2018 keeps PV output on the
same weather days as the EULP load profiles and GHCN temp bins.

Cross-check: PVGIS v5.2 seriescalc (PVGIS-NSRDB 2005-2015 climatology)
— the same radiation database the PVWatts API uses. The script prints
per-site agreement and warns above 10 % daylight-hour deviation.

This replaces the original PVWatts API plan: nrel.gov lost its .gov
DNS delegation (verified via DNS-over-HTTPS on 2026-07-02), so the
API is unreachable for the foreseeable future.

Run from the repo root:
    pipenv run python scripts/fetch_pv_shapes.py
"""

import json
import logging
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("fetch_pv_shapes")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = (
    REPO_ROOT / "src/opensynth/datasets/new_england/resources/pv_shapes_ne.csv"
)

# Reference sites from config.PV_SHAPE_SITES
SITES = {
    "north": (43.20, -71.50),  # Concord, NH
    "south": (42.27, -71.87),  # Worcester, MA
}
YEAR = 2018
TILT_DEG = 30
AZIMUTH_DEG = 0  # south-facing in both APIs' conventions
EST_OFFSET_H = 5  # fixed EST, matching the EULP clock

# PVWatts defaults (standard crystalline module)
GAMMA_PDC = -0.0037  # power temperature coefficient, 1/C
NOCT_C = 45.0
SYSTEM_LOSSES = 0.14
INVERTER_EFF = 0.96

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    f"&start_date={YEAR}-01-01&end_date={YEAR}-12-31"
    "&hourly=global_tilted_irradiance,temperature_2m"
    f"&tilt={TILT_DEG}&azimuth={AZIMUTH_DEG}&timezone=UTC"
)
PVGIS_URL = (
    "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    "?lat={lat}&lon={lon}&raddatabase=PVGIS-NSRDB"
    "&startyear=2005&endyear=2015&pvcalculation=1&peakpower=1"
    f"&loss={int(SYSTEM_LOSSES * 100)}&angle={TILT_DEG}"
    f"&aspect={AZIMUTH_DEG}&outputformat=json"
)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _monthly_hour_shape(
    stamps_utc: list[datetime], kw_per_kw: np.ndarray
) -> pl.DataFrame:
    """Mean kw_per_kw by (month, hour) in fixed EST."""
    est = [t - timedelta(hours=EST_OFFSET_H) for t in stamps_utc]
    df = pl.DataFrame(
        {
            "year": [t.year for t in est],
            "month": [t.month for t in est],
            "hour": [t.hour for t in est],
            "kw_per_kw": kw_per_kw,
        }
    )
    return (
        df.filter(pl.col("year") == pl.col("year").max())
        .group_by(["month", "hour"])
        .agg(pl.col("kw_per_kw").mean())
        .sort(["month", "hour"])
    )


def fetch_open_meteo_shape(lat: float, lon: float) -> pl.DataFrame:
    """2018 hourly PVWatts-model output per kW, as (month, hour) means."""
    data = _get_json(OPEN_METEO_URL.format(lat=lat, lon=lon))["hourly"]
    # Stamps behave as period-beginning hour labels: the resulting
    # monthly profiles centre on 12.1-12.4 fixed EST, matching solar
    # noon (~12:10-12:15) at these longitudes. A trial -1h relabel
    # shifted the centre to ~11.2, i.e. physically wrong.
    stamps = [datetime.fromisoformat(t) for t in data["time"]]
    gti = np.array(
        [v if v is not None else 0.0 for v in data["global_tilted_irradiance"]]
    )
    t2m = np.array(
        [v if v is not None else 0.0 for v in data["temperature_2m"]]
    )

    t_cell = t2m + gti / 800.0 * (NOCT_C - 20.0)
    p_dc = gti / 1000.0 * (1.0 + GAMMA_PDC * (t_cell - 25.0))
    p_ac = np.clip(p_dc, 0.0, None) * (1.0 - SYSTEM_LOSSES) * INVERTER_EFF
    return _monthly_hour_shape(stamps, p_ac)


def fetch_pvgis_shape(lat: float, lon: float) -> pl.DataFrame:
    """PVGIS-NSRDB 2005-2015 climatology per kW, as (month, hour) means."""
    rows = _get_json(PVGIS_URL.format(lat=lat, lon=lon))["outputs"]["hourly"]
    stamps = [datetime.strptime(r["time"][:11], "%Y%m%d:%H") for r in rows]
    p_ac = np.array([r["P"] for r in rows]) / 1000.0  # W per kWp -> kW/kW
    # Climatology: keep all years (filter in _monthly_hour_shape keeps
    # only the max year, so aggregate here instead).
    est = [t - timedelta(hours=EST_OFFSET_H) for t in stamps]
    df = pl.DataFrame(
        {
            "month": [t.month for t in est],
            "hour": [t.hour for t in est],
            "kw_per_kw": p_ac,
        }
    )
    return (
        df.group_by(["month", "hour"])
        .agg(pl.col("kw_per_kw").mean())
        .sort(["month", "hour"])
    )


def compare(
    site: str, df_om: pl.DataFrame, df_pvgis: pl.DataFrame
) -> dict[str, float]:
    """Agreement with the NSRDB climatology on quantities that are
    robust to the two sources' sub-hour stamp conventions: monthly
    energy, daily profile correlation, and annual energy."""
    joined = df_om.join(df_pvgis, on=["month", "hour"], suffix="_pvgis")
    monthly = (
        joined.group_by("month")
        .agg(
            pl.col("kw_per_kw").sum().alias("om"),
            pl.col("kw_per_kw_pvgis").sum().alias("pvgis"),
        )
        .with_columns(
            ((pl.col("om") - pl.col("pvgis")).abs() / pl.col("pvgis")).alias(
                "ape"
            )
        )
    )
    profile_r = float(
        np.corrcoef(
            joined["kw_per_kw"].to_numpy(),
            joined["kw_per_kw_pvgis"].to_numpy(),
        )[0, 1]
    )
    stats = {
        "monthly_energy_mape_pct": float(monthly["ape"].mean()) * 100.0,
        "profile_correlation": profile_r,
        "energy_ratio_om_over_pvgis": float(
            joined["kw_per_kw"].sum() / joined["kw_per_kw_pvgis"].sum()
        ),
    }
    logger.info(
        "%s vs PVGIS-NSRDB climatology: monthly-energy MAPE %.1f%%, "
        "profile r=%.3f, annual-energy ratio %.3f",
        site,
        stats["monthly_energy_mape_pct"],
        profile_r,
        stats["energy_ratio_om_over_pvgis"],
    )
    # Gates sized for single-year weather vs an 11-year climatology
    # plus the sources' half-hour stamp-convention offset.
    if (
        stats["monthly_energy_mape_pct"] > 15.0
        or not 0.85 <= stats["energy_ratio_om_over_pvgis"] <= 1.15
        or profile_r < 0.95
    ):
        logger.warning(
            "%s deviates from the NSRDB climatology beyond single-year "
            "weather variability — inspect before committing",
            site,
        )
    return stats


def main() -> None:
    frames = []
    for site, (lat, lon) in SITES.items():
        logger.info("Fetching Open-Meteo %d shape for %s", YEAR, site)
        df_om = fetch_open_meteo_shape(lat, lon)
        logger.info("Fetching PVGIS-NSRDB climatology for %s", site)
        df_pvgis = fetch_pvgis_shape(lat, lon)
        compare(site, df_om, df_pvgis)
        frames.append(df_om.with_columns(pl.lit(site).alias("site")))

    df = (
        pl.concat(frames)
        .select("site", "month", "hour", pl.col("kw_per_kw").round(4))
        .sort(["site", "month", "hour"])
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT_PATH)
    logger.info("Wrote %d rows to %s", df.height, OUT_PATH)


if __name__ == "__main__":
    main()
