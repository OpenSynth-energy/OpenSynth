# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the New England synthetic AMI dataset pipeline.

Encodings, data source locations, and augmentation parameters for
training a Faraday model on NREL End-Use Load Profiles (ResStock
AMY2018) for the six New England states.
"""

NE_STATES = ["CT", "MA", "ME", "NH", "RI", "VT"]

STATE_ENCODING = {state: i for i, state in enumerate(NE_STATES)}

# Archetype from EULP metadata column `in.geometry_building_type_recs`.
# Multi-family 2-4 and 5+ unit values collapse to one archetype.
# Keys verified against NH release-2 metadata (2026-07-01).
ARCHETYPE_ENCODING = {
    "Single-Family Detached": 0,
    "Single-Family Attached": 1,
    "Multi-Family with 2 - 4 Units": 2,
    "Multi-Family with 5+ Units": 2,
    "Mobile Home": 3,
}

# Heating fuel from EULP metadata column `in.heating_fuel`.
HEATING_FUEL_ENCODING = {
    "Electricity": 0,
    "Natural Gas": 1,
    "Fuel Oil": 2,
    "Propane": 3,
    "Other Fuel": 4,
    "None": 4,
}

# Daily mean temperature bins (degrees C), 5-degree steps.
# np.digitize(temp_c, TEMP_BIN_EDGES_C) yields integer bins 0-9:
# bin 0 = below -15C, bin 9 = above 25C.
TEMP_BIN_EDGES_C = [-15, -10, -5, 0, 5, 10, 15, 20, 25]

# Fixed conditioning label order shared by preprocessing, the data
# module, and conditional sampling. class_dim = len(FEATURE_COLS).
FEATURE_COLS = [
    "state",
    "archetype",
    "heating_fuel",
    "has_ev",
    "has_pv",
    "month",
    "dayofweek",
    "temp_bin",
]

# One GHCN-Daily station per state (airport stations). 2018 is the
# EULP AMY weather year. All six IDs verified to return 2018
# TMAX/TMIN via the NCEI data service (2026-07-01).
GHCN_STATIONS = {
    "CT": "USW00014740",  # Hartford Bradley Intl
    "MA": "USW00094746",  # Worcester Regional
    "ME": "USW00014764",  # Portland Intl Jetport
    "NH": "USW00014745",  # Concord Municipal
    "RI": "USW00014765",  # Providence T.F. Green
    "VT": "USW00014742",  # Burlington Intl
}

WEATHER_YEAR = 2018

# EULP ResStock AMY2018 release 2 on the OEDI data lake (public,
# anonymous HTTPS). Both key layouts verified: 1,200 building
# timeseries files downloaded June 2026, metadata HEAD-checked
# 2026-07-01.
EULP_BASE_URL = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2024/"
    "resstock_amy2018_release_2"
)
EULP_TIMESERIES_URL = (
    EULP_BASE_URL
    + "/timeseries_individual_buildings/by_state/upgrade=0/"
    + "state={state}/{bldg_id}-0.parquet"
)
EULP_METADATA_URL = (
    EULP_BASE_URL
    + "/metadata_and_annual_results/by_state/state={state}/parquet/"
    + "{state}_baseline_metadata_and_annual_results.parquet"
)

# RECS 2020 state microdata (public domain). 56 MB CSV.
RECS_URL = (
    "https://www.eia.gov/consumption/residential/data/2020/csv/"
    "recs2020_public_v7.csv"
)
# Destination of the RECS microdata inside the data directory; shared
# by the downloader and every loader.
RECS_CSV_RELPATH = "raw/new_england/recs/recs2020_public_v7.csv"

# NCEI data service for GHCN-Daily summaries (anonymous, CSV).
GHCN_DATA_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    "?dataset=daily-summaries&stations={station}"
    "&startDate={year}-01-01&endDate={year}-12-31"
    "&dataTypes=TMAX,TMIN&format=csv"
)

# Stratified sampling of the training corpus.
BUILDINGS_PER_STATE = 300
MIN_BUILDINGS_PER_CELL = 5
SAMPLING_SEED = 42

# EV charging augmentation, cold-climate L2 parameters.
EV_CHARGE_KW = 7.2
EV_ARRIVAL_MEAN_HOUR = 19.0
EV_ARRIVAL_STD_HOURS = 1.25
EV_ARRIVAL_WINDOW_HOURS = (16.0, 23.0)
EV_ENERGY_MEAN_KWH = 9.5
EV_ENERGY_STD_KWH = 2.5
EV_WINTER_ENERGY_FACTOR = 1.3  # applies when temp_bin <= 2
EV_DAILY_CHARGE_PROB = 0.75

# PV augmentation: normalized hourly shapes (PVWatts simplified model
# on Open-Meteo ERA5 2018 irradiance, PVGIS-NSRDB cross-checked; see
# scripts/fetch_pv_shapes.py) precomputed for one northern and one
# southern reference site, committed under resources/. Sizes drawn
# uniformly from PV_SIZES_KW.
PV_SIZES_KW = [4, 6, 8]
PV_SHAPE_SITES = {
    "north": "Concord, NH",
    "south": "Worcester, MA",
}
PV_STATE_SHAPE = {
    "CT": "south",
    "MA": "south",
    "ME": "north",
    "NH": "north",
    "RI": "south",
    "VT": "north",
}
