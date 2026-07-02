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
# Verify raw metadata values on first download before relying on this.
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

# One GHCN-Daily station per state (airport stations, hourly-quality
# daily summaries). 2018 is the EULP AMY weather year.
# Verify IDs against the GHCN-D station inventory on first download.
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
# anonymous HTTPS). The per-building timeseries key layout below is
# verified: 1,200 building files (200 per NE state) were downloaded
# with it in June 2026. The metadata key layout still needs a
# `curl -I` check on first use.
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

# PV augmentation: normalized PVWatts hourly shapes precomputed for
# one northern and one southern reference site, committed as small
# resource CSVs. Sizes drawn uniformly from PV_SIZES_KW.
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
