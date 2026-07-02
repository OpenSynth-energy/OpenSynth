# Dataset Card — OpenSynth New England Synthetic Residential AMI v1.0

> **Status: RELEASE CANDIDATE.** All evaluation and generation fields
> are final. Remaining `{TBD:...}` markers are release logistics only:
> Zenodo DOI, citation, release date.

## Summary

1,000 synthetic New England residential electricity meter profiles at
15-minute resolution for one full year (calendar 2018 weather), in the
**net-load convention**: homes flagged with rooftop PV have generation
subtracted from consumption, so readings go negative during export. No
real household appears anywhere in the pipeline — the generative model
is trained entirely on NREL's *simulated* ResStock building stock.

Produced with the LF Energy OpenSynth implementation of Centre for Net
Zero's Faraday model (conditional VAE + Gaussian Mixture Model),
extended with closed-form conditional GMM sampling so profiles can be
generated for chosen labels (state, dwelling type, heating fuel, DER
ownership, calendar, daily temperature).

| | |
|---|---|
| Homes | 1,000 |
| Interval | 15 minutes (96 readings/day) |
| Span | 365 days, weather year 2018, fixed EST clock |
| Rows | ~35.0 M interval readings |
| States | CT, MA, ME, NH, RI, VT (RECS household-weighted) |
| Convention | Net load (PV homes export; negatives are real) |
| DOI | `{TBD:zenodo-doi}` |

## Files

| File | Contents |
|---|---|
| `ne_synthetic_1000homes.parquet` | `home_id`, `timestamp`, `kwh` (long format) |
| `ne_synthetic_1000homes.csv.gz` | Same data, CSV for non-parquet consumers |
| `ne_synthetic_1000homes_metadata.csv` | One row per home: `home_id`, `state_postal`, `state`, `archetype`, `heating_fuel`, `has_pv`, `has_ev`, `magnitude_scale` |
| `DATASET_CARD.md` | This card |

### Timestamps

Period-beginning, fixed EST year-round (no DST transitions), matching
the NREL EULP convention. `2018-01-15 17:00` covers 17:00–17:15 EST.

### Label encodings

| Label | Values |
|---|---|
| `state` | CT 0, MA 1, ME 2, NH 3, RI 4, VT 5 |
| `archetype` | 0 single-family detached, 1 single-family attached, 2 multi-family, 3 mobile home |
| `heating_fuel` | 0 electricity, 1 natural gas, 2 fuel oil, 3 propane, 4 other/none |
| `has_pv`, `has_ev` | 0/1 |

## How the data was generated

1. **Training corpus** — 1,800 buildings (300 per state) drawn from
   NREL End-Use Load Profiles (`resstock_amy2018_release_2`, upgrade
   0), stratified within each state over archetype × heating fuel to
   the RECS 2020 weighted joint distribution. The exact building IDs
   are pinned in a committed manifest. 657,000 building-days →
   492,750 train / 164,250 holdout, split **by building**.
2. **DER augmentation** — PV and home-EV-charging flags assigned from
   RECS 2020 state-level adoption shares (104 PV, 16 EV homes of
   1,800). PV homes get PVWatts-model generation netted out
   (system sizes 4/6/8 kW; shapes from Open-Meteo ERA5 actual-2018
   irradiance, cross-checked against the PVGIS-NSRDB climatology —
   see *PV shapes* below). EV homes get cold-climate L2 charging
   sessions injected (7.2 kW, arrival ~N(19:00, 1.25 h), 9.5 ± 2.5
   kWh/session, ×1.3 energy below −5 °C, charge probability 0.75).
3. **Conditioning** — each daily profile carries 8 integer labels:
   state, archetype, heating_fuel, has_ev, has_pv, month, dayofweek,
   and a 10-bin daily-mean-temperature label from the state's GHCN
   airport station (5 °C bins, −15 °C to +25 °C).
4. **Model** — Faraday VAE (input 96, latent 16, class_dim 8) trained
   150 epochs on all training profiles; GMM with
   200 (sweep winner: k=100 gave +16.0 % annual MPE, k=400 gave −16.4 %) components fitted over the 24-dim joint
   (latent, label) space. Final VAE training loss
   0.840.
5. **Sampling** — per home: labels drawn from the RECS 2020
   within-state joint distribution (states weighted by RECS household
   weights, DER flags by state adoption shares); then one day sampled
   per 2018 calendar date by conditioning the GMM on (labels, month,
   dayofweek, that state's **real 2018 temp-bin trajectory**) in
   closed form and decoding. Cold snaps and heat waves therefore land
   on the correct dates, coherently across homes in the same state.
6. **Magnitude calibration** — day-independent sampling collapses
   per-home annual variance (365 draws per home average toward the
   segment mean, losing persistent home-level identity: synthetic
   annual std ~1,100 kWh vs 6,354 in the real corpus). Each home
   therefore receives one persistent scale factor, drawn from the
   empirical distribution of relative annuals (home ÷ segment mean)
   among real training-split homes of its archetype × heating-fuel
   segment (pooled fallback below 20 homes; clamped to [0.25, 4.0]).
   The factor is recorded per home as `magnitude_scale` in the
   metadata file. Shapes, peak timing and DER signatures are
   unaffected; annual-kWh realism is restored (see Evaluation).

### PV shapes

The PVWatts API was unreachable during production (the `nrel.gov`
domain lost its `.gov` DNS delegation), so shapes are computed from
Open-Meteo ERA5 hourly tilted irradiance and temperature for actual
2018, through the PVWatts simplified model (NOCT cell temperature,
−0.37 %/°C, 14 % system losses, 96 % inverter efficiency; 30° tilt,
south-facing; Concord NH serves ME/NH/VT, Worcester MA serves
CT/MA/RI). Cross-check against the PVGIS-NSRDB 2005–2015 climatology
(the same radiation database PVWatts uses): profile correlation
≈ 0.975, monthly-energy MAPE 11–12 %, annual-energy ratio 0.91–0.93 —
within single-year weather variability. Capacity factors 13.9–14.0 %.
Using actual 2018 keeps PV output on the same weather days as the
load profiles and temperature labels.

## Evaluation

Following the LF Energy evaluation framework for synthetic smart meter
data (Chai et al. 2024, *Defining 'Good'*, arXiv:2407.11785):
**fidelity / utility / privacy**. Full numbers and executed notebooks:
`notebooks/new_england/ne_evaluation.ipynb`.

### Fidelity — external references (regional realism)

| Criterion | Threshold | Result |
|---|---|---|
| Mean annual kWh per home vs RECS 2020 weighted NE mean | ±5 % | **+4.1 %**, KS 0.076, std 4,660 vs RECS 4,706 (released dataset, after magnitude calibration) — pass. Model-level before calibration: −5.9 %, KS 0.302 |
| Winter aggregate peak timing vs ISO-NE hourly demand | ≤ 60 min | **0 min** (synthetic 19:00 = ISO-NE 19:00) — pass |
| Summer aggregate peak timing vs ISO-NE hourly demand | ≤ 60 min | 60 min (synthetic 18:00 vs ISO-NE 17:00) — pass at tolerance |
| Seasonal mean profile shape vs EULP holdout | Pearson r > 0.85 | r = 0.95–0.99 across seasons — pass |
| EV evening signature (16–23 h energy-share uplift, EV vs non-EV) | uplift present | 0.466 vs 0.354 evening energy share — pass |
| PV midday signature (net-load depression, PV vs non-PV twin) | depression present | −0.30 kWh/15 min midday depression — pass |

Caveats: the ISO-NE reference is 2019 (EIA's hourly archive starts
there) against a 2018 weather year — seasonal peak timing is stable
across adjacent years; ISO-NE system load includes commercial and
industrial demand, so the comparison is about *timing*, not magnitude.

### Fidelity — distributional (synthetic vs holdout)

Computed with the OpenSynth evaluation tooling on year-long per-home
series (real holdout label trajectories):

| Metric | Result |
|---|---|
| Seasonal peak-count distributions, pairwise KS | KS 0.27 (fall) – 0.77 (spring); see limitations |
| Autocorrelation-coefficient distributions (hour/half-day/day/week lags), pairwise KS | KS 0.65–0.76 (hour–week lags), 0.27 (half-year); see limitations |

Note: Faraday generates days independently; multi-day coherence enters
only through the shared real temperature trajectory, so long-lag ACF
agreement is expected to be the weakest fidelity result. This is a
known architectural property, not a data defect.

### Utility — Train-on-Synthetic, Test-on-Real

Season classification (Dec–May vs Jun–Nov) from daily profiles,
logistic regression, tested on real holdout days:

| Trained on | Accuracy |
|---|---|
| Synthetic data | 0.572 |
| Real training data | 0.575 |
| Absolute delta (utility measure) | **0.002** |

### Privacy

**This dataset contains no personal data.** The generative model was
trained exclusively on NREL ResStock *simulated* buildings — physics
model outputs, not real meters — so there is no household to
re-identify and membership-inference / reconstruction attacks have no
victim. The OpenSynth privacy tooling (MIA, reconstruction, outlier
poisoning, per the framework paper) was therefore not run for v1; it
becomes mandatory if this pipeline is ever retrained on real AMI data.

## Known limitations

- **EV homes are rare and weakly learned.** RECS 2020 home-charging
  shares are 0.6–1.7 %, giving 13 EV homes in the training mix. v1 is
  deliberately RECS-honest; users studying EV load should treat the
  `has_ev` signature as directional. A documented EV-oversampled
  training mix is the planned alternative if demand warrants.
- **Single weather year.** All profiles reflect 2018 New England
  weather; the dataset does not span inter-annual variability.
- **Days are conditionally independent.** Within-home day-to-day
  persistence beyond what the shared temperature trajectory and the
  per-home magnitude scale induce (e.g., vacations, occupancy
  streaks) is not modelled. The distributional ACF/peak-count KS
  results above quantify this.
- **Rare extreme readings.** 0.003 % of readings exceed 25 kW and 24
  of 35 M exceed 50 kW — sampled outlier days amplified by large
  magnitude scales. Winsorize if your application is sensitive to
  implausible single-interval peaks.
- **Simulated ground truth.** Fidelity is measured against ResStock
  simulations and public statistics (RECS, ISO-NE), not against real
  New England meter data, which is not publicly available — that gap
  is the reason this dataset exists.
- **PV shapes are model-based** (PVWatts model on reanalysis
  irradiance), not measured generation; see cross-check above.
- Aggregations across many homes reproduce *population* statistics;
  individual synthetic homes are not forecasts of any real home.

## Reproducibility

Everything is pinned in the OpenSynth repo (branch
`feature/new-england`, upstream PR [OpenSynth-energy/OpenSynth#89](https://github.com/OpenSynth-energy/OpenSynth/pull/89)):

```
pipenv run python app/app.py get-ne-data --timeseries    # ~13 GB, manifest-pinned
pipenv run python scripts/fetch_pv_shapes.py             # PV shapes + cross-check
pipenv run python app/app.py preprocess-ne-data \
    --pv_shapes src/opensynth/datasets/new_england/resources/pv_shapes_ne.csv
pipenv run python scripts/train_ne_faraday.py            # hours, CPU
pipenv run python app/app.py generate-ne-dataset        # 1,000-home dataset
```

Building manifest: `data/raw/new_england/building_manifest.csv`
(committed hash sha256 `967d21338d77f652…`). All RNGs seeded; DER
assignment, splits and sampling are deterministic given the manifest.

## Sources and licenses

| Source | Use | License |
|---|---|---|
| NREL End-Use Load Profiles (`resstock_amy2018_release_2`) | Training profiles | CC BY 4.0 |
| EIA RECS 2020 microdata | Stratification, DER shares, validation | Public domain (US Gov) |
| NOAA GHCN-Daily | Temperature labels | Public domain (US Gov) |
| EIA hourly demand (ISO-NE) | Peak-timing validation | Public domain (US Gov) |
| Open-Meteo (ERA5) | PV irradiance/temperature | CC BY 4.0 |
| PVGIS (EU JRC) | PV cross-check only | Free reuse with attribution |

**This dataset**: CC BY 4.0 (proposed, pending sign-off).
Code: Apache-2.0 (OpenSynth).

## Citation

```
{TBD:citation}  (Zenodo DOI on release)
```

Please also cite the Faraday model (Chai & Chadney 2024) and the
evaluation framework (Chai et al. 2024, arXiv:2407.11785) when using
the evaluation results.

## Changelog

- **v1.0** (`{TBD:release-date}`) — initial release: 1,000 homes,
  RECS-honest DER mix, net-load convention.
