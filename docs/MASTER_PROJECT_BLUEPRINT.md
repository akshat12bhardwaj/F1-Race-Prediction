# ML-Based Formula 1 Race Outcome Prediction
## Master Project Blueprint (v1.0)

| | |
|---|---|
| Course | CSE20232 Machine Learning, BML Munjal University, AY 2026–27 (Batch 2024, Sem 5) |
| Component | End-semester Project (40%), CO1 + CO2 + CO3 |
| Document date | 2026-09-25 |
| Status | Design approved for implementation; no code written yet |

### How to read this document

Every important statement is tagged with one of these labels:

| Tag | Meaning |
|---|---|
| **[CONFIRMED]** | Checked against the course handout or with live API requests on 2026-09-25 |
| **[DESIGN]** | A decision this blueprint proposes. It can be changed, but only on purpose and with a written reason |
| **[ASSUMPTION]** | Believed true but not yet verified. The step where it will be checked is named |
| **[OPTIONAL]** | Only if time allows. The project is complete without it |

Scope is tiered as **MUST** (the project fails without it), **SHOULD** (expected for a strong grade) and **COULD** (extra).

---

### 0. What the Course Handout Requires [CONFIRMED]

The handout is the authoritative source. These are its project-relevant facts and how this blueprint meets each one.

| Handout requirement | How this blueprint satisfies it |
|---|---|
| End-semester Project = **40%**, "during the last two weeks of the course" (semester runs 27 Jul – 30 Nov 2026) | Timeline in §42 finishes all technical work by mid-November |
| "A comprehensive Project report ... along with the coding solutions; followed by presentation, demonstration & Viva" | §38 report, §39 presentation and demo, §40 viva bank, §41 deliverables |
| "There would be **one pre-submission**" | Checkpoints 1–7 (§33) are designed to be the pre-submission package |
| "implement and show results of the proposed solution and perform a **comparative analysis with different available algorithms**" | Model ladder (§23–25), experiment plan (§29), comparison tables (§28) |
| "A student needs to spend approximately **20 hours**" | Scope tiers and hour budget in §42 |
| "A **separate assessment** will be conducted for evaluating the solution provided by **each student**" | Every team member must be able to explain the whole pipeline (§42, team note) |
| Course overview: students implement "some of the most popular ML algorithms **from scratch**" and do "hyperparameter tuning and model selection in **statistically robust** ways" | SHOULD item: logistic regression from scratch (E11). Season-based CV and bootstrap confidence intervals (§27–28) |
| CO1: classification + regression with feature engineering and feature selection | §20, §21, §23, §24 |
| CO2: appropriate models for a real-world problem | Problem framing (§3–5), leakage-safe design (§19) |
| CO3: evaluate on various performance parameters | §28: accuracy, P/R/F1, ROC-AUC, PR-AUC, confusion matrix, MAE/RMSE/R², bootstrap CI, calibration |

**The handout does not specify these, so ask the instructor:** whether the project is individual or team, the report length and format, the pre-submission date and contents, the rubric weights, and whether external APIs are acceptable data sources.

---

## 1. Executive Summary

This project builds a supervised ML system that predicts Formula 1 Grand Prix outcomes at a clearly defined moment: **after the starting grid is known and before the race starts**. It combines driver form, constructor form, qualifying and grid position, circuit history, championship context and weather. It does not rely on one statistic.

Key design decisions:

1. **Two data sources only.** Jolpica F1 API (Ergast-compatible) supplies all racing data. The Open-Meteo Historical Weather API supplies weather. No Kaggle dataset is needed for the core pipeline.
2. **One row = one driver in one Grand Prix.** There are about 25,900 rows for 1950–2025 [CONFIRMED count, §10].
3. **Separate models per target.** Finishing-position regression is primary. Winner and podium classification are secondary. DNF classification is optional.
4. **A strict pre-race feature rule.** Every feature is computed from races strictly before the race being predicted, and automated leakage tests check this.
5. **Temporal evaluation.** Train on 1950–2021 with season-based expanding-window CV for tuning. Validate on 2022. Test on 2023–2025, which is touched once.
6. **Strong naive baselines.** The "predict the grid order" and "pole-sitter wins" heuristics are baselines every ML model must beat. This keeps the project honest.
7. **Model ladder.** Heuristic → linear models (Logistic/Linear/Ridge/Lasso) → Decision Tree → bagging (Random Forest) → boosting (HistGradientBoosting, XGBoost optional). The comparison is the result.
8. **Race-level outputs.** Driver-level predictions are turned into race-level predictions: win probabilities normalised per race, a predicted podium, and a predicted finishing order. Uncertainty is reported with bootstrap confidence intervals.
9. **Honest weather.** Historical reanalysis is not a forecast. Weather is treated as a "perfect-forecast proxy" and tested with an ablation experiment.

---

## 2. Project Title

**ML-Based Formula 1 Race Outcome Prediction: A Leakage-Safe, Multi-Factor Comparison of Linear, Tree-Based and Ensemble Models**

(Short title for slides: *Predicting F1 Race Outcomes with Machine Learning*)

---

## 3. Problem Statement

Given the information available just before a Formula 1 Grand Prix starts, predict for every driver on the grid:

- their **finishing position** (regression),
- whether they will **win** (binary classification),
- whether they will finish on the **podium** (binary classification),
- optionally, whether they will **not finish** (DNF, binary classification).

Aggregate these into a race-level prediction: a win-probability table, a predicted podium and a predicted finishing order. Evaluate them against what actually happened, using only historical data that would have been available at prediction time.

The problem is hard because outcomes depend on many interacting factors, some unpredictable (crashes, mechanical failures, safety cars, rain). The positive classes are rare (1 winner per ~20 drivers). The sport's regulations and competitive order drift over 75 years.

---

## 4. Motivation

- **Real-world, data-rich problem.** 75 years of structured public data suit a course project on supervised learning (CO2).
- **Tests a common claim.** "Pole position decides the race" is widely believed. We can measure how much ML adds beyond grid position.
- **Covers the whole course.** It needs regression and classification, feature engineering and selection, ensembles, tuning, CV, bootstrap and ROC (CO1, CO3).
- **Teaches methodology.** Sports data makes leakage and temporal drift easy to see and explain, and a viva will probe exactly these points.
- **Stakeholder interest.** Teams, broadcasters, fantasy-league players and analysts use similar pre-race models.

---

## 5. Research Question

**Main RQ:** *Can the outcome of a Formula 1 Grand Prix be predicted more accurately by combining major pre-race factors (driver form, constructor form, qualifying/grid position, circuit history, championship context and weather) than by relying on a single statistic such as starting grid position?*

Sub-questions:

| ID | Sub-question | Answered by |
|---|---|---|
| RQ1 | Do multi-factor ML models beat the grid-only heuristic on winner, podium and position prediction? | E1–E5 |
| RQ2 | Which factor groups (grid, driver form, constructor form, circuit, weather) contribute most? | E6, E7 |
| RQ3 | Do ensembles (bagging, boosting) beat linear models and single trees, and by a statistically meaningful margin? | E3, E5, E10 |
| RQ4 | Does weather add predictive value, and how much of any gain depends on using observed rather than forecast weather? | E6 |
| RQ5 | How well do models trained on historical eras generalise to recent seasons? Does a modern-only training window help? | E9 |

Hypotheses (to be tested, not assumed):

- **H1:** Grid position is the strongest single predictor.
- **H2:** Adding constructor and driver form improves podium and position prediction beyond grid alone.
- **H3:** Boosted trees give the best ranking metrics, but the margin over logistic regression is small.
- **H4:** Weather has a small average effect but matters in wet races.

---

## 6. Objectives

1. Build a reproducible, cached data pipeline from the Jolpica and Open-Meteo APIs for 1950–2025.
2. Create a validated master dataset (one row per driver per Grand Prix) with documented cleaning rules.
3. Engineer at least 25 pre-race features, each passing a written leakage audit and automated tests.
4. Implement at least 3 classification and at least 4 regression algorithms plus heuristic baselines.
5. Tune the main models with season-based cross-validation.
6. Compare models with multiple metrics and bootstrap 95% confidence intervals on an untouched 2023–2025 test set.
7. Produce calibrated per-race win and podium probabilities and a predicted finishing order.
8. Deliver the report, code, presentation, live demo and viva preparation.

---

## 7. Scope

| In scope (MUST/SHOULD) | Optional (COULD) | Out of scope |
|---|---|---|
| Championship Grands Prix 1950–2025 (Indianapolis 500 excluded, §17) | Sprint result as a feature (2021+) | Deep learning / sequence models |
| Pre-race features from Jolpica | Constructor lineage mapping (renamed teams) | Lap-by-lap race simulation |
| Open-Meteo weather (reanalysis) | Circuit characteristics table (manual) | Real-time live-timing ingestion |
| Regression: finishing position | DNF classifier | Betting or financial use |
| Classification: winner, podium | XGBoost | Practice-session pace (needs extra source; Future Work) |
| Linear, tree, bagging, boosting models | KNN / SVM comparison runs | Complex deployment (servers, Docker, cloud) |
| Season-based CV, tuning, bootstrap CI | Logistic regression from scratch (SHOULD, per handout ethos) | Training on 2026 data |
| CLI prediction demo | Streamlit demo UI | |

---

## 8. Course Alignment

| Handout topic | Where it appears in this project |
|---|---|
| Supervised learning; regression vs classification | Three separate targets (§14) |
| Cost function, gradient descent | Logistic regression from scratch, trained by gradient descent on log-loss (E11) |
| Under/overfitting, bias–variance | Tree depth validation curve; learning curve (§22, E8) |
| Cost-sensitive models | `class_weight='balanced'` for rare winner/podium classes (§23) |
| Linear Regression, Ridge, Lasso | Regression ladder; Lasso also used for feature selection (§21, §24) |
| Logistic Regression, SVM, Decision Tree, KNN | Classification ladder; SVM/KNN as optional comparison (§23) |
| Ensemble: Decision Forest, Bagging, Boosting | Random Forest, HistGradientBoosting, optional XGBoost (§25) |
| Probability, Bayes rule, maximum likelihood | Logistic regression as MLE; probability calibration; normalising probabilities within a race (§30) |
| PCA / Kernel PCA | COULD: PCA on correlated rolling-form features for linear models (E12) |
| Clustering, anomaly detection | Future Work: circuit clustering; anomaly flags for suspicious results (§37) |
| ANN | Deliberately excluded; justified in §37 and the viva bank |
| Accuracy, CI, confusion matrix, P/R, ROC, R², CV, bootstrap, tuning | §26–28 |
| Model deployment | Saved `joblib` pipeline and command-line prediction script (§30) |

| Course outcome | Evidence in the deliverables |
|---|---|
| CO1 | Feature engineering (§20), feature selection (§21), classification and regression models |
| CO2 | End-to-end system for a real prediction problem with a leakage-safe design |
| CO3 | Multi-metric, CI-backed comparison tables and plots |

---

## 9. System Architecture

```
┌──────────────────────────── DATA SOURCES ────────────────────────────┐
│  Jolpica F1 API (races, results,         Open-Meteo Archive API      │
│  qualifying, sprint, circuits)           (hourly reanalysis weather) │
└───────────────┬──────────────────────────────────┬───────────────────┘
                ▼                                  ▼
   data/raw/jolpica/**.json            data/raw/openmeteo/**.json    ← RAW (immutable, cached)
                └───────────────┬──────────────────┘
                                ▼
                     DATA VALIDATION  → outputs/tables/validation_report.csv
                                ▼
                     DATA CLEANING    → data/interim/*.csv (flat tables)
                                ▼
                     DATA INTEGRATION → data/interim/master_base.csv
                                ▼
                     FEATURE ENGINEERING (past-only) → data/processed/master_features.csv
                                ▼
                     LEAKAGE AUDIT (automated tests)  → outputs/tables/leakage_audit.csv
                                ▼
                     EDA  → outputs/figures/eda_*.png
                                ▼
          TEMPORAL SPLIT: train ≤2021 │ val 2022 │ test 2023–2025
                                ▼
                     FEATURE SELECTION (train folds only)
                                ▼
     ┌─────────────── CLASSIFICATION ───────────────┐  ┌──────── REGRESSION ────────┐
     │ winner / podium (/ DNF)                      │  │ finishing position         │
     │ heuristic → LogReg → DT → RF → HGB (→ XGB)   │  │ heuristic → Lin/Ridge/Lasso│
     └──────────────────────┬───────────────────────┘  │ → DT → RF → HGB (→ XGB)    │
                            │                          └─────────────┬──────────────┘
                            ▼                                        ▼
              HYPERPARAMETER TUNING (season-fold CV) → VALIDATION (2022)
                            ▼
              CALIBRATION + RACE-LEVEL DECODING + MODEL COMPARISON
                            ▼
              FINAL MODELS (refit ≤2022) → ONE-TIME TEST (2023–2025) + bootstrap CI
                            ▼
              RACE PREDICTION: win % table │ predicted podium │ predicted order
                            ▼
              VISUALISATION → REPORT → SLIDES → DEMO → VIVA
```

---

## 10. Data Sources

### 10.1 Core sources

| Source | Role | Required? | Coverage (verified 2026-09-25) | Why it is needed |
|---|---|---|---|---|
| **Jolpica F1 API** `https://api.jolpi.ca/ergast/f1/` | All racing data: schedule, circuits with lat/long, results (grid, position, status, points), qualifying, sprint | **MUST** | 1950 onward. 1,172 races listed including the 2026 schedule. 26,181 result rows including 308 from 2026 **[CONFIRMED]** | Only free, structured, Ergast-compatible source that covers 1950–2025 in one schema. Provides the targets and most features |
| **Open-Meteo Historical Weather API** `https://archive-api.open-meteo.com/v1/archive` | Hourly weather at circuit coordinates on race day | **SHOULD** (the project still works without weather; weather is tested by ablation) | Returns data for 1950-05-13 at Silverstone and 2024-03-02 at Sakhir **[CONFIRMED]** | Jolpica has no weather. Open-Meteo is free, needs no key, and covers the whole period through ERA5 reanalysis |

Derived size **[CONFIRMED from API totals; recount at Step 7]**: 1950–2025 has about **1,149 races** (1,172 − 23 scheduled 2026 rounds) and about **25,873 result rows** (26,181 − 308).

### 10.2 Kaggle datasets considered and why they are not used

| Dataset | Decision | Reason |
|---|---|---|
| Formula 1 World Championship 1950–2020 | Not used | Same Ergast origin as Jolpica but stops at 2020 |
| Formula 1 Complete Dataset 1950–2026 | Not used; **emergency fallback** only if Jolpica becomes unavailable | Duplicates Jolpica; unknown curation; 2026 data is out of scope |
| Formula 1 Weather Info 1950–2024 | Not used as a feature source. **COULD** use to cross-check our rain flag on a sample | Provenance unclear, ends 2024; Open-Meteo is reproducible |
| F1 Stint Data with Aggression Scores | Rejected | Stints and tyres are in-race information, so using them would leak |
| Formula 1 Pit Stop Dataset | Rejected | Pit stops are in-race. Jolpica already has them (2011+) if ever needed for EDA |
| Formula 1 Race Data and Telemetry | Rejected | Telemetry is in-race and very large |
| Formula 1 Official Practice Sessions Data | Future Work | Practice pace is valid pre-race information. Jolpica provides session **dates only**, not timings **[CONFIRMED]**, so this would be the one justified extra source. Coverage is modern-era only, so it is left out of the core scope |
| Formula 1 Race Dataset for Predictive Analysis | Rejected | Pre-engineered features with unknown leakage properties |

**Principle:** add a source only when a specific, missing, pre-race variable is needed and cannot be derived from Jolpica or Open-Meteo.

---

## 11. API Architecture

### 11.1 Jolpica endpoints to use

| Endpoint | URL pattern | Fields taken | Verified facts |
|---|---|---|---|
| Season schedule | `/{season}/races.json` | season, round, raceName, date, time, Circuit(circuitId, name, lat, long, locality, country) | `time` (UTC, e.g. `15:00:00Z`) is **present from 2005** and **null through 2004** **[CONFIRMED]**. Modern races also list FirstPractice/Qualifying/Sprint session dates and times, but no results |
| Race results | `/{season}/results.json?limit=100&offset=k` | number, position, positionText, points, Driver(driverId, code, dateOfBirth, nationality), Constructor(constructorId, name), grid, laps, status, Time, FastestLap | **Max page size = 100** (asking for 1000 returned 100) **[CONFIRMED]** |
| Qualifying | `/{season}/qualifying.json?limit=100&offset=k` | position, driverId, constructorId, Q1, Q2, Q3 | **1994 onward** (1993 returns 0 rows). Q1 only until 2004; Q1+Q2 in 2005; Q1/Q2/Q3 from 2006 **[CONFIRMED]** |
| Sprint | `/{season}/sprint.json` | position, points, grid | Exists from 2021 **[CONFIRMED]** |
| Driver/constructor standings | `/{season}/{round}/driverStandings.json` | position, points, wins | Returns standings **after** that round **[CONFIRMED]**. No constructor standings before 1958 **[CONFIRMED]**. **Not used as features** (see §20.4); used only to validate our derived standings |
| Pit stops, laps | `/{season}/{round}/pitstops.json` | — | Pit stops from 2011 **[CONFIRMED]**. In-race, so **forbidden as features**. EDA only, if at all |

**Pagination gotcha [DESIGN]:** Ergast-style APIs paginate on result rows, not races. One race's `Results` array can be split across two pages. The parser must **concatenate results by (season, round)** across pages.

**Rate limits [ASSUMPTION, per Jolpica documentation; verify at Step 2]:** about 4 requests/second burst and about 500 requests/hour unauthenticated. Design: sleep ≥ 1 s between calls, exponential back-off on HTTP 429/5xx, and cache every response to disk so a request is never repeated.

**Request budget estimate:** schedules 76 + results ~300 + qualifying ~160 + sprint ~10 ≈ **550 requests**, about 1–2 hours in total and resumable. Downloading official standings for every round would add about 2,300 requests. This is a second reason to derive standings ourselves.

### 11.2 Open-Meteo access

- Endpoint: `https://archive-api.open-meteo.com/v1/archive`
- Parameters: `latitude`, `longitude` (from Jolpica `Circuit.Location`), `start_date = end_date = race_date`, `hourly=temperature_2m,relative_humidity_2m,precipitation,rain,cloud_cover,wind_speed_10m,wind_direction_10m,weather_code`, `timezone=auto`.
- Response **[CONFIRMED]**: `hourly.time[]` plus one array per variable, `utc_offset_seconds`, `elevation`, and **grid-snapped coordinates**: the request for 26.0325, 50.5106 returned 26.0457, 50.5142.
- One request per race gives about 1,149 requests. **[ASSUMPTION, per Open-Meteo docs]** The free tier allows about 600/min, 5,000/h and 10,000/day, so a throttled run at 1 request/s takes about 20 minutes.
- No API key is needed. Use is non-commercial academic.

### 11.3 Client design (to implement later in `src/api_client.py`)

```
get_json(url, cache_path):
    if cache_path exists → load and return      # never re-download
    for attempt in 1..5:
        response = GET url (timeout 30 s, User-Agent "F1-ML-course-project")
        if 200 → save raw bytes to cache_path; return json
        if 429 or 5xx → sleep(2**attempt) and retry
        else → raise with URL + status
    log failure to data/raw/_failed_requests.csv
```

Each downloaded file is logged in `data/raw/_manifest.csv` with URL, timestamp, HTTP status, byte size and row count.

---

## 12. Dataset Design

### 12.1 Row granularity **[DESIGN]**

**One row = one driver participating in one Grand Prix.** Primary key: `(race_id, driver_id)`.

Why this structure:

1. **It matches the prediction unit.** Every question (win? podium? position?) is asked per driver per race.
2. **It supports all targets in one table.** Winner, podium, position and DNF are all columns derived from the same row.
3. **Standard models can use it.** Scikit-learn estimators expect independent feature vectors.
4. **Race-level answers are easy to produce.** Group by `race_id` to normalise probabilities and rank predictions within a race.
5. **Leakage-safe feature building is simple.** Sort by date, group by driver or constructor, and shift by one race.

Alternatives considered:

| Alternative | Why it was not chosen |
|---|---|
| One row per race (predict the winner ID) | Hundreds of classes, and the set of drivers changes every season |
| Pairwise rows (driver A beats driver B?) | Elegant for ranking, but about 20× more rows and harder to explain in a 20-hour project. Kept as Future Work |
| One row per driver-season | Too coarse; cannot predict individual races |

**Important consequence:** rows in the same race are not independent. Exactly one winner per race, and positions are a permutation. This is handled by race-level decoding (§30) and by bootstrapping over races, not rows (§28).

### 12.2 Modelling population **[DESIGN]**

A row enters the modelling dataset if the driver was **on the starting grid**, which is known at the prediction point. Exclude:

- the Indianapolis 500 (1950–1960). It is present in the data **[CONFIRMED for 1950]** but it is a different racing series in practice. Filter on `raceName == "Indianapolis 500"`, **not** on circuitId. **[ASSUMPTION]** circuitId `indianapolis` is also used for the 2000–2007 US Grands Prix; verify at Step 7.
- non-starters: did not qualify, did not pre-qualify, withdrawn, did not start (`positionText` in {F, W} or status in {Did not qualify, Did not prequalify, Did not start, Withdrew}).

**[ASSUMPTION]** The `positionText` codes are R = retired, D = disqualified, E = excluded, W = withdrawn, F = failed to qualify, N = not classified. R and W are **[CONFIRMED]** in samples; verify the others at Step 7.

These rows are **kept in interim tables**, so history is complete, but they are excluded from training and evaluation rows.

---

## 13. Master Dataset Schema

Categories: **A** = directly available from source · **B** = derived by feature engineering · **C** = optional · **D** = needs an additional data source.

### 13.1 Identifiers and metadata (not model inputs)

| Column | Type | Cat. | Source | Notes |
|---|---|---|---|---|
| `race_id` | str | B | season + round, e.g. `2024_01` | Primary key part 1 |
| `season` | int | A | races | |
| `round` | int | A | races | |
| `race_date` | date | A | races.date | Sorting key |
| `race_start_utc` | datetime | A / B | races.time (2005+) or imputed 14:00 local (≤2004) | Imputed flag stored |
| `race_name` | str | A | races.raceName | |
| `circuit_id` | str | A | Circuit.circuitId | 78 circuits all-time **[CONFIRMED]** |
| `circuit_lat`, `circuit_lon`, `country` | float/str | A | Circuit.Location | Used for weather lookup |
| `driver_id` | str | A | Driver.driverId | Primary key part 2 |
| `constructor_id` | str | A | Constructor.constructorId | |
| `driver_dob`, `driver_nationality` | date/str | A | Driver | |

### 13.2 Raw pre-race inputs

| Column | Type | Cat. | Available | Notes |
|---|---|---|---|---|
| `grid_position` | int | A | 1950+ | **[ASSUMPTION]** 0 means pit-lane start; recode to `field_size + 1`, flag `pit_lane_start`. Verify at Step 7 |
| `qualifying_position` | int | A | **1994+ only** | Missing before 1994, so impute with grid and add `quali_missing` flag |
| `q1_time`, `q2_time`, `q3_time` | str | A | 1994+ / 2005+ / 2006+ | Parse to seconds |
| `sprint_position` | int | C | 2021+ | Sprint is held before the GP, so it is pre-race |

### 13.3 Derived pre-race features (Category B unless stated)

| Column | Definition (always past races only) |
|---|---|
| `field_size` | Number of starters in this race (known from grid) |
| `grid_pct` | (grid − 1) / (field_size − 1); an era-comparable grid measure |
| `front_row` | grid ≤ 2 |
| `quali_best_s` | Best of available Q1/Q2/Q3 in seconds |
| `quali_gap_to_pole_pct` | (quali_best_s / pole_best_s − 1) × 100 |
| `teammate_grid_diff` | grid − teammate grid (same race, pre-race) [C] |
| `driver_age_days` | race_date − dob |
| `driver_career_starts` | Count of prior starts |
| `is_rookie` | career_starts < 10 [DESIGN threshold] |
| `driver_prev_finish` | Finishing position in the driver's previous race |
| `driver_last3_avg_finish`, `driver_last5_avg_finish`, `driver_last10_avg_finish` | Rolling mean of past finishing positions |
| `driver_career_avg_finish` | Expanding mean of past finishing positions |
| `driver_last10_avg_finish_pct` | Rolling mean of past finishing percentile (era-comparable) |
| `driver_win_rate`, `driver_podium_rate` | Expanding (and last-20) share of past wins/podiums |
| `driver_dnf_rate_last10` | Share of DNFs in last 10 starts |
| `driver_avg_positions_gained_last10` | Mean of (grid − finish) over last 10 |
| `driver_points_before_race` | Season points from rounds < current (incl. sprint if collected) |
| `driver_points_share_before_race` | points / leader's points (era-comparable) |
| `driver_champ_pos_before_race` | Rank of points_before_race within season |
| `driver_wins_season_before_race` | Season wins so far |
| `constructor_last5_avg_finish` | Rolling mean over past 5 races of the team's mean car finish |
| `constructor_win_rate_last20`, `constructor_podium_rate_last20` | Past team race-level success |
| `constructor_dnf_rate_last10` | Team reliability proxy |
| `constructor_points_before_race`, `constructor_points_share_before_race`, `constructor_champ_pos_before_race` | Derived season standings (valid pre-1958 as our own construct) |
| `constructor_prev_season_rank` | Final derived rank in previous season |
| `driver_circuit_starts`, `driver_circuit_avg_finish` | Past visits of this driver to this circuit |
| `constructor_circuit_avg_finish` | Past team results at this circuit |
| `circuit_pole_win_rate` | Share of past races at this circuit won from pole |
| `circuit_dnf_rate` | Share of DNFs in past races at circuit |
| weather features | §13.4 |
| `circuit_type`, `circuit_length_km`, `n_corners` | **D**: manual table from Wikipedia; COULD |
| `practice_best_gap_pct` | **D**: practice timing (e.g. FastF1); Future Work |
| `constructor_lineage_id` | **C**: manual map for renamed teams (see §17) |

### 13.4 Weather columns (A from Open-Meteo, aggregated as B)

| Column | Aggregation over race window [start, start + 2 h] |
|---|---|
| `wx_temp_mean_c` | mean temperature_2m |
| `wx_humidity_mean` | mean relative_humidity_2m |
| `wx_precip_sum_mm`, `wx_rain_sum_mm` | sum precipitation / rain |
| `wx_wind_mean_kmh` | mean wind_speed_10m |
| `wx_cloud_mean` | mean cloud_cover |
| `wx_code_max` | max weather_code (worst condition) |
| `wx_is_wet` | 1 if precip_sum ≥ 0.5 mm OR any hour weather_code ≥ 51 (threshold set after the sanity check at Step 6) |
| `wx_prestart_*` | Same aggregations over [start − 3 h, start − 1 h] (used for the weather ablation, E6) |
| `wx_time_imputed` | 1 if start time was imputed (≤2004) |

Wind direction is collected but **not** used in v1 (it is circular and has low expected value).

### 13.5 Target columns (never model inputs)

`target_finish_position`, `target_winner`, `target_podium`, `target_dnf`, plus helper `finish_pct`. Defined in §14.

### 13.6 Outcome columns kept only for history building (never direct inputs)

`points`, `laps`, `status`, `positionText`, `race_time_ms`, `fastest_lap_*`. They are used only to compute *past* features and the targets.

---

## 14. Target Variables

| Target | Type | Definition | Priority |
|---|---|---|---|
| `target_finish_position` | Regression (integer 1..N) | Jolpica `position` field. It gives every starter a position, including retirements ordered by laps completed **[ASSUMPTION: verify ordering at Step 7]** | **PRIMARY** |
| `target_winner` | Binary | 1 if `position == 1` else 0 | SECONDARY |
| `target_podium` | Binary | 1 if `position <= 3` else 0 | SECONDARY |
| `target_dnf` | Binary | 1 if `positionText in {"R","N"}`. See rule below | OPTIONAL |
| Constructor performance | Analysis | Aggregate driver predictions per team per race (expected points, best predicted position) and compare with actual | OPTIONAL |

**DNF definition [DESIGN].** Status text is not consistent across eras. 2010 uses granular labels (`Engine`, `Collision`, `+1 Lap`). 2023–2024 use coarse labels (`Finished`, `Lapped`, `Retired`, `Did not start`, `Disqualified`) **[CONFIRMED]**. There are 136 distinct status values all-time **[CONFIRMED]**. So:

- Primary rule: `positionText` (R = retired, N = not classified) → DNF = 1.
- Finished = status is `Finished`, `Lapped`, or matches `^\+\d+ Laps?$` → DNF = 0.
- Disqualified (D/E): DNF = 0, flagged `is_dsq`, and excluded from DNF model training only.
- Cross-check: counts of DNF by rule A (positionText) vs rule B (status) per season must match within 1%; mismatches are listed in the validation report.

**Class balance (approximate):** winner ≈ 1/field_size ≈ 5% in the modern era (lower in the 1950s with 30+ car fields). Podium ≈ 15%. DNF varies from about 10% (modern) to over 40% (1950s–80s).

**One model per target [DESIGN].** Train separate models rather than one model doing everything, because:

1. Each target has a different loss (squared/absolute error vs log-loss) and different class balance.
2. Features that matter for winning (front-row grid, top-team form) differ from those for mid-field positions or DNF (reliability).
3. Separate models are easier to evaluate and explain in the viva.

**Consistency check:** winner and podium predictions can also be *derived* from the regression ranking. Experiment E4 compares "dedicated classifier" against "derived from regression". The better method on validation supplies the podium in the final output.

---

## 15. Data Collection Plan

| Order | What | Endpoint | Raw file pattern | Est. requests |
|---|---|---|---|---|
| 1 | Season schedules 1950–2025 | `/{season}/races.json?limit=100` | `data/raw/jolpica/races/{season}.json` | 76 |
| 2 | Results 1950–2025 | `/{season}/results.json?limit=100&offset=k` | `data/raw/jolpica/results/{season}_{offset:04d}.json` | ~300 |
| 3 | Qualifying 1994–2025 | `/{season}/qualifying.json?...` | `data/raw/jolpica/qualifying/{season}_{offset:04d}.json` | ~160 |
| 4 | Sprint 2021–2025 (SHOULD) | `/{season}/sprint.json?...` | `data/raw/jolpica/sprint/{season}_{offset:04d}.json` | ~10 |
| 5 | Standings sample (validation only) | `/{season}/{round}/driverStandings.json` | `data/raw/jolpica/standings_sample/{season}_{round}.json` | ~20 |
| 6 | Weather per race | Open-Meteo archive | `data/raw/openmeteo/{season}_{round:02d}.json` | ~1,149 |

Rules:

- Collect seasons as a loop over `1950..2025`. **2026 is never downloaded into training folders.** An optional 2026 live-demo pull goes to `data/raw/jolpica_2026_demo/`.
- Each script is **idempotent**: rerunning it skips existing files.
- Page loop: `offset += 100` until `offset >= total`.

---

## 16. Data Storage Plan

```
data/
├── raw/            # IMMUTABLE. Exact API bytes. Never edited by hand. Git-ignored (re-creatable)
│   ├── jolpica/{races,results,qualifying,sprint,standings_sample}/
│   ├── openmeteo/
│   ├── _manifest.csv          # url, file, fetched_at_utc, status, n_rows
│   └── _failed_requests.csv
├── interim/        # Flat, cleaned, typed tables (CSV); one per entity
│   ├── races.csv  results.csv  qualifying.csv  sprint.csv  weather_race.csv
│   └── master_base.csv        # integrated, no engineered features yet
├── processed/      # Model-ready
│   ├── master_features.csv    # all rows 1950–2025 with features + targets
│   ├── train.csv  val.csv  test.csv   # split snapshots (for reproducibility)
│   └── feature_list_v1.json   # the exact selected feature names
└── external/       # (COULD) manual tables: circuit_meta.csv, constructor_lineage.csv
```

Rules: raw is read-only. Every downstream file can be rebuilt by scripts. CSV is chosen for transparency (about 26k rows is small). A `data/DATA_DICTIONARY.md` documents every column. Record the Python and library versions in `requirements.txt` with pinned versions.

---

## 17. Data Cleaning Plan

| Issue | Rule [DESIGN] | Verified at |
|---|---|---|
| Numbers stored as strings (`"1"`, `"26"`) | Cast `position, grid, points, laps, round, season` to numeric | Step 8 |
| `positionText` non-numeric codes | Keep. Use them for DNF/DSQ/DNS flags | Step 7 |
| Indianapolis 500 1950–1960 | Drop from modelling rows and from driver history | Step 7 |
| Non-starters (F, W, DNS) | Keep in interim; exclude from modelling population | Step 8 |
| Grid = 0 | Pit-lane start → `field_size + 1`, flag set. If it seems to mean "unknown" in old races, set NaN | Step 7 |
| Qualifying times `"1:29.179"` / empty / `"DNF"` | Parse to seconds; invalid → NaN | Step 8 |
| Missing qualifying before 1994 | NaN + `quali_missing = 1`; impute (grid) inside the model pipeline only | Step 11 |
| Missing race `time` ≤2004 | Impute 14:00 local; `wx_time_imputed = 1` | Step 6 |
| Shared drives (1950s, several drivers credited for one car) **[ASSUMPTION: present as separate rows]** | Keep rows; history uses each driver's own row | Step 7 |
| Duplicate (race_id, driver_id) | Must be 0 after cleaning; if a shared drive creates duplicates, keep the row with the best position and log it | Step 7 |
| Constructor renames (e.g. Toro Rosso → AlphaTauri → RB; Force India → Racing Point → Aston Martin) **[ASSUMPTION: IDs differ per name]** | v1: accept the reset. COULD: `constructor_lineage.csv` mapping | Step 11 |
| Half/double points races, dropped scores (pre-1991) | Derived standings differ from official. Document; use `points_share` and ranks | Step 11 |
| Status taxonomy change (2023+ coarse) | Use `positionText` for DNF (§14) | Step 7 |
| Weather grid snapping | Store returned lat/lon and the distance to the circuit | Step 6 |

Every rule is logged with the number of affected rows in `outputs/tables/cleaning_log.csv`.

---

## 18. Data Integration Plan

| Join | Keys | Cardinality | Check |
|---|---|---|---|
| results ⟵ races | (season, round) | many-to-1 | Every result finds a race; 0 orphans |
| results ⟵ qualifying | (season, round, driver_id) | 1-to-0..1 | Match rate ≥ 95% for 1994+ starters. List non-matches |
| results ⟵ sprint | (season, round, driver_id) | 1-to-0..1 | Only sprint weekends |
| results ⟵ weather_race | (season, round) | many-to-1 | 100% of races have weather or an explicit failure flag |
| (COULD) ⟵ circuit_meta | circuit_id | many-to-1 | |

Post-integration assertions (Step 9):

- `master_base` has as many rows as the cleaned results, and `(race_id, driver_id)` is unique.
- Exactly one `position == 1` per race (modelling population).
- Row count per season matches the API totals minus the documented exclusions.

---

## 19. Data Leakage Prevention and Prediction-Time Feature Audit

### 19.1 Prediction point **[DESIGN]**

> **T₀ = the moment the official starting grid is final, before the formation lap.**
> A feature is allowed if and only if its value was knowable at T₀ **for that specific race**.

### 19.2 Forbidden as inputs for the race being predicted

| Forbidden item | Why |
|---|---|
| Current race finishing position / `positionText` | It is the target |
| Current race points | A direct function of the target |
| Current race pit stops, laps, lap times, fastest lap, race time | Generated during the race |
| Current race laps completed | Reveals DNF and finishing order |
| Current race status / DNF | Outcome |
| Current race incidents, safety cars, overtakes | During the race |
| Standings *after* the current round (API `/{season}/{round}/driverStandings`) | Includes this race's points |
| Season-total stats of the current season (e.g. final season wins) | Includes future races |
| Target-encoded driver/team IDs fitted on the full data | Encode outcomes, including test outcomes |
| Scalers, imputers or selectors fitted on all data | Leak test-set distribution |
| Race-window weather, treated as if it were a forecast | Observed during the race. Allowed only with the labelling in §19.5 |

### 19.3 Feature audit table

| Feature | Available pre-race? | Past races only? | Could it include the current race by accident? | How we verify |
|---|---|---|---|---|
| grid_position | Yes (grid published) | n/a (current, pre-race) | No | — |
| qualifying_position / times | Yes (qualifying before race) | n/a | No | Qualifying date < race date in schedule |
| sprint_position | Yes (Saturday) | n/a | No | Sprint date < race date |
| field_size | Yes (grid) | n/a | Only if computed from finishers: **compute from starters** | Unit test |
| driver_prev_finish / lastN / career avg | Yes | Must be | **Yes**, if `.shift(1)` is forgotten | Tests T1, T2 |
| win/podium/DNF rates | Yes | Must be | Yes, same risk | T1, T2 |
| points / champ pos before race | Yes | Rounds < current only | **Yes**, if the API "after round r" standings are used | T1; compare with API standings at r − 1 |
| constructor rolling features | Yes | Must be | **Yes**, via teammate: aggregate per (team, race) **then** shift | T1, T2 |
| constructor_prev_season_rank | Yes | Previous season | Yes, if the current season is used | T1 |
| circuit history features | Yes | Previous visits only | Yes, if current visit included | T1, T2 |
| circuit_pole_win_rate | Yes | Previous races at circuit | Yes | T1 |
| driver_age / experience | Yes | n/a | No | — |
| weather (race window) | **No**: observed | n/a | By definition it is race-time | Labelled "oracle forecast proxy"; ablation E6 |
| weather (pre-start window) | Observed before T₀, but not a forecast of the race | n/a | No | E6 |

### 19.4 Implementation rules

1. Sort by `(race_date, race_id)` before any rolling computation.
2. All rolling/expanding features follow **aggregate → shift(1) → roll**. Example: `groupby('driver_id')[x].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())`.
3. Team features: first build a team-race table (one row per constructor per race), then shift within the team, then merge back.
4. Standings: cumulative season sum **minus the current race** (use shift) and rank within `(season, round)`.
5. **Split before fit.** Every imputer, scaler and selector lives inside a scikit-learn `Pipeline` fitted only on training folds.
6. No driver, constructor or circuit ID encodings as features (they overfit and cannot generalise to new drivers).
7. **Test lock.** `test.csv` (2023–2025) is loaded by one notebook only (`12_final_evaluation`), after all choices are frozen.

### 19.5 Automated leakage tests (`tests/test_leakage.py`)

| ID | Test | Pass condition |
|---|---|---|
| T1 | **Recompute test.** For 50 random rows, rebuild each feature from `master_base[race_date < this race_date]` only | Identical values |
| T2 | **Perturbation test.** Change the target and outcome columns of one race, rebuild features | Features of that race unchanged; later races change |
| T3 | **Cold-start test.** A driver's first career race | All driver-history features NaN |
| T4 | **Suspicion test.** Single-feature AUC for winner | No non-grid feature has AUC > 0.95 (would suggest leakage) |
| T5 | **Split test.** No race_id in more than one split; max(train date) < min(val date) < min(test date) | True |
| T6 | **Column denylist.** Model feature list contains none of: position, points, laps, status, time, fastest, pit, positionText, target_* | True |

### 19.6 Weather: honesty statement

Three different things are called "race weather":

1. **Actual race-day weather**: what really happened at the track (track sensors, official reports). **Not available to us.**
2. **Historical reanalysis** (what Open-Meteo archive provides): a physics-model reconstruction of past weather on a grid of about 9–25 km, blended with observations. It approximates (1) at grid-cell scale.
3. **Pre-race forecast**: what forecasters predicted before T₀. **Not what we use.**

Using (2) for the race window is equivalent to assuming a **perfect forecast**, which overstates what is knowable at T₀. So: (a) all weather results are labelled "oracle-forecast proxy"; (b) experiment E6 compares no weather, pre-start window weather, and race-window weather; (c) the report must not claim the weather features were available before the race. **[ASSUMPTION]** Open-Meteo also offers a Historical *Forecast* API (archived forecasts, about 2022+). It is a Future Work option for genuine forecast features.

---

## 20. Feature Engineering

### 20.1 Procedure

1. Start from `master_base` (full history including rows excluded from modelling, so driver history is complete).
2. Compute helper columns: `finish_pct = (position − 1)/(field_size − 1)`, `is_win`, `is_podium`, `is_dnf`, `positions_gained = grid − position`.
3. Compute each feature family with **aggregate → shift → roll** (§19.4).
4. Filter to the modelling population.
5. Save `master_features.csv` and run leakage tests T1–T6.

### 20.2 Feature families

| Family | Features | Rationale |
|---|---|---|
| **Grid/qualifying** | grid_position, grid_pct, front_row, pit_lane_start, qualifying_position, quali_gap_to_pole_pct, quali_missing, (teammate_grid_diff) | Starting position is the strongest known predictor. Gap-to-pole measures pace better than rank |
| **Driver recent form** | prev_finish, last3/5/10 avg finish, last10 avg finish_pct, avg positions gained last 10 | Short-term momentum |
| **Driver long-term** | career avg finish, win rate, podium rate, dnf rate last 10, career starts, age, is_rookie | Skill and experience |
| **Championship context** | points before race, points share, champ pos, season wins so far | Title fights, within-season hierarchy |
| **Constructor form** | last5 avg finish, win/podium rate last 20, dnf rate last 10, points share, champ pos, prev season rank | The car usually matters more than the driver in F1 |
| **Circuit history** | driver_circuit_starts, driver_circuit_avg_finish, constructor_circuit_avg_finish, circuit_pole_win_rate, circuit_dnf_rate | Track-specific skill; how processional a track is |
| **Weather** | §13.4 | Rain mixes up the order |
| **(COULD) Sprint** | sprint_position | Strong same-weekend signal (2021+) |
| **(D) Circuit meta** | street circuit flag, length, corners | Overtaking difficulty |

### 20.3 Cold start and missing values

- Rookies, new teams and first visits to a circuit have NaN history features. Tree models (HistGradientBoosting) handle NaN natively. For other models: `SimpleImputer(strategy="median", add_indicator=True)` inside the pipeline.
- Rolling windows use `min_periods=1`, plus a companion `*_n` count feature where useful, so the model knows how much history there is.

### 20.4 Era comparability

Points systems changed many times, so raw points are not comparable across eras. Use **points share** (points / leader's points) and **ranks**. Field size changed (about 34 in the 1950s to 20 today), so use **percentile features** (`finish_pct`, `grid_pct`) alongside raw positions.

### 20.5 Expected first feature set v1

About 30 features: 8 grid/quali, 7 driver form, 6 driver long-term, 4 championship, 7 constructor, 5 circuit, 8 weather. Feature selection (§21) reduces this.

---

## 21. Feature Selection

All steps run **on training data only** (inside CV folds where results feed model choice).

1. **Rule-based filter (hard):** leakage denylist (T6), identifiers, near-constant columns (variance ≈ 0).
2. **Redundancy filter:** Pearson/Spearman correlation on training data. For pairs with |r| > 0.95 (e.g. last5 vs last10 avg), keep the more interpretable one.
3. **Embedded selection:** Lasso (regression) and L1-logistic (classification). Features with zero coefficients across most folds are candidates to drop.
4. **Model-based importance:** permutation importance on the **validation** season (2022), for the best tree ensemble. Impurity importance is shown only for comparison, since it is biased toward high-cardinality features.
5. **Feature-group ablation (E7):** drop one family at a time and measure the change in validation metrics.
6. Freeze `feature_list_v1.json`. Record which features were dropped and why in `outputs/tables/feature_selection_log.csv`.

(COULD) Recursive Feature Elimination with season-fold CV (`RFECV`) as a comparison.

---

## 22. EDA Plan

All figures go to `outputs/figures/` as PNG (300 dpi) with the prefix `eda_`.

| # | Analysis | Plot | Question it answers |
|---|---|---|---|
| 1 | Missing values by column and era | Heatmap (column × decade) | Where is data incomplete (qualifying pre-1994, start time pre-2005)? |
| 2 | Races and entrants per season | Line/bar | How did calendar and field size change? |
| 3 | Finishing-position distribution | Histogram | Is the target uniform? Retirement effects? |
| 4 | **Grid vs finish** | Scatter + jitter + LOWESS; heatmap of P(finish = k \| grid = g) | How much does the grid determine the result? |
| 5 | Pole-to-win rate by decade | Bar | Has the grid become more decisive? |
| 6 | Qualifying gap vs finish (1994+) | Scatter | Is pace gap more informative than rank? |
| 7 | Driver win and podium rates (top 20) | Horizontal bar | Dominance of few drivers |
| 8 | Constructor win share by decade | Stacked bar | Team dominance eras |
| 9 | DNF rate by decade + top DNF causes (pre-2023 granular labels) | Line + bar | Reliability trend; DNF class balance |
| 10 | Circuit "processionality": mean abs(grid − finish) per circuit | Bar | Which tracks allow overtaking? |
| 11 | Weather distributions; wet vs dry races | Hist + box | How common are wet races? |
| 12 | Wet vs dry: grid–finish correlation | Grouped bar | Does rain shuffle the order? |
| 13 | Feature correlation matrix (train split) | Heatmap | Redundancy for §21 |
| 14 | Feature distributions by target (winner vs not) | Box/violin | Separability |
| 15 | Outliers: extreme positions gained, grid = 0, very large fields | Box + table | Data quality |
| 16 | Class imbalance summary | Bar | Motivates class weights and PR-AUC |
| 17 | Temporal drift: mean feature values by season | Line | Motivates temporal split and window experiment |

The EDA is summarised in a findings table: "Observation → modelling implication".

---

## 23. Classification Methodology

**Tasks:** (1) winner, (2) podium, (3) optional DNF. Separate models for each.

### 23.1 Model ladder

| Level | Model | Why included |
|---|---|---|
| Baseline 0 | `DummyClassifier(strategy="prior")` | Floor: predicts base rate |
| Baseline 1 | **Heuristic:** winner = grid 1; podium = grid ≤ 3. Probabilities = historical P(win \| grid) from the training data | **The bar to beat.** Answers RQ1 |
| Linear | **Logistic Regression** (L2; also L1 for selection), scaled features, `class_weight` tuned | Interpretable coefficients and probabilistic output; core course algorithm |
| Linear (from scratch, SHOULD) | NumPy logistic regression with batch gradient descent | Handout's "from scratch" ethos; shows the cost function and gradient descent |
| Tree | **Decision Tree** | Non-linear, interpretable rules. Shows overfitting (depth curve) |
| Bagging | **Random Forest** | Reduces tree variance; importance analysis |
| Boosting | **HistGradientBoostingClassifier** | Strong on tabular data, handles NaN, fast on about 24k rows |
| Boosting (COULD) | XGBoost | Only if time permits. Adds a dependency with little conceptual gain over HGB. Include if the examiner expects it |
| Optional comparison | KNN (k tuned) | Course coverage. Expected to struggle with imbalance and mixed scales |
| Optional comparison | SVM (linear via `LinearSVC` + calibration, or RBF on a ≥2006 subset) | Course coverage. RBF-SVM is O(n²), so limit the data. No native probabilities |

### 23.2 Class imbalance handling

- Report PR-AUC and F1, not accuracy alone. With a 5% base rate, "never predicts a winner" is 95% accurate.
- Cost-sensitive learning: `class_weight="balanced"` as a tuned option.
- **No SMOTE by default.** Synthetic driver-race rows break the race structure. Race-level decoding (exactly one winner per race) handles imbalance more naturally.
- Threshold: for row-level P/R/F1, choose the threshold on validation (maximise F1). Race-level decoding uses argmax / top-k, so it needs no threshold.

### 23.3 Race-level decoding

- Winner: normalise within a race, `p̂ᵢ = pᵢ / Σⱼ pⱼ`. Predicted winner = argmax.
- Podium: top 3 drivers by podium probability.

---

## 24. Regression Methodology

**Target:** `target_finish_position` (primary). Optional variant: `finish_pct`, for cross-era training.

| Level | Model | Why |
|---|---|---|
| Baseline 0 | `DummyRegressor(mean)` | Floor |
| Baseline 1 | **Heuristic:** predicted position = grid position | Bar to beat |
| Linear | **Linear Regression** | Reference linear model; coefficients interpretable |
| Regularised | **Ridge** (L2), **Lasso** (L1) | Collinear rolling features; Lasso doubles as feature selection |
| Tree | **Decision Tree Regressor** | Non-linearity; bias–variance demo |
| Bagging | **Random Forest Regressor** | Variance reduction |
| Boosting | **HistGradientBoostingRegressor** (loss = squared_error; optionally absolute_error) | Expected best |
| COULD | XGBoost Regressor | As above |

**Race-level decoding:** continuous predictions are **ranked within each race** to produce integer positions 1..N. Ties are broken by win probability, then grid. Report both raw-prediction MAE and ranked-position MAE.

---

## 25. Ensemble Learning

| Ensemble type | Implementation | What it demonstrates |
|---|---|---|
| Bagging | Random Forest (bootstrap rows + random feature subsets) | Averaging many high-variance trees lowers variance. Out-of-bag score as a free estimate |
| Boosting | HistGradientBoosting (sequential trees fitting residuals/gradients) | Reduces bias. Needs a learning rate, a number of iterations and early stopping |
| (COULD) Soft voting | `VotingClassifier` of LogReg + RF + HGB | Whether mixing different model families helps |
| (COULD) Stacking | `StackingClassifier` with a logistic meta-learner, using **season-fold** CV internally | Higher leakage risk; only if time permits |

Emphasis: ensembles are judged **against** the heuristic and linear baselines, with confidence intervals. "More complex" is not assumed to be "better".

---

## 26. Hyperparameter Tuning

**Strategy [DESIGN]:** `RandomizedSearchCV` (n_iter = 20–30) with **season-based CV splits** (§27). `GridSearchCV` is used only for small spaces (Ridge/Lasso alpha, LogReg C, tree depth).
**Scoring:** classification = `average_precision` (PR-AUC; robust to imbalance, threshold-free), with log-loss reported. Regression = `neg_mean_absolute_error`.
**Fixed seed:** `random_state=42` everywhere.

| Model | Parameters and values | Search |
|---|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10}; penalty ∈ {l2, l1 (solver=saga)}; class_weight ∈ {None, balanced} | Grid |
| Ridge | alpha ∈ logspace(−3, 3, 13) | Grid |
| Lasso | alpha ∈ logspace(−4, 0, 9) | Grid |
| Decision Tree | max_depth ∈ {3, 5, 8, 12, None}; min_samples_leaf ∈ {1, 5, 20, 50}; (class_weight) | Grid |
| Random Forest | n_estimators ∈ {200, 500}; max_depth ∈ {None, 8, 16}; min_samples_leaf ∈ {1, 5, 20}; max_features ∈ {sqrt, 0.5}; class_weight ∈ {None, balanced_subsample} | Random (20) |
| HistGradientBoosting | learning_rate ∈ {0.03, 0.1, 0.2}; max_leaf_nodes ∈ {15, 31, 63}; min_samples_leaf ∈ {20, 50, 100}; l2_regularization ∈ {0, 0.1, 1}; max_iter = 500 with early_stopping | Random (25) |
| XGBoost (COULD) | n_estimators, learning_rate, max_depth ∈ {3, 5, 7}, subsample, colsample_bytree, scale_pos_weight | Random (25) |
| KNN (opt.) | n_neighbors ∈ {5, 15, 51, 101}; weights ∈ {uniform, distance} | Grid |

**Leakage-safe tuning:** the whole `Pipeline(imputer → scaler → model)` is tuned, so preprocessing is re-fitted inside each fold. The 2022 validation and 2023–2025 test seasons are never inside the tuning CV.

**Time budget:** about 24k rows × 5 folds × about 25 configs is minutes per model on a laptop. Use `n_jobs=-1`.

---

## 27. Cross-Validation and Temporal Split

### 27.1 Why a random split is wrong

A random `train_test_split` would (a) put 2024 races in training while testing on 2019, i.e. **training on the future**; (b) split the rows of one race across train and test, so teammates' and rivals' same-race rows would inform each other; (c) overstate performance because consecutive seasons are similar (temporal autocorrelation). Performance must be measured the way the model will be used: trained on the past, predicting the future.

### 27.2 Why plain `TimeSeriesSplit` is not enough

`TimeSeriesSplit` splits by row index. It can cut a race (about 20 rows) in half. We need a **grouped, season-level expanding window**, implemented as a small custom generator in `src/splits.py` that yields (train_idx, val_idx) and is passed to `cv=` in scikit-learn.

### 27.3 Recommended split **[DESIGN; confirm at Step 13 after inspecting the data]**

```
Tuning CV (inside training data, expanding window by season):
  Fold 1: train 1950–2016 → validate 2017
  Fold 2: train 1950–2017 → validate 2018
  Fold 3: train 1950–2018 → validate 2019
  Fold 4: train 1950–2019 → validate 2020  (short COVID season)
  Fold 5: train 1950–2020 → validate 2021
Hold-out validation: train 1950–2021 → validate 2022   (model selection, calibration, thresholds)
Final test (ONCE):   refit on 1950–2022 → test 2023–2025 (≈70 races, ≈1,400 rows)
```

Reasoning:

- **Test = 2023–2025.** These are the three most recent complete seasons (about 70 races) and the same regulation era (2022 ground-effect rules), so the test is a clean "near future" check. 2026 is excluded: the season is incomplete **[CONFIRMED: 308 of about 460 results]** and brings major new technical rules.
- **Validation = 2022.** It is the first season of the current regulations, so selecting on it favours models that handle a rule change. The weakness is that 22 races is small, so calibration may use out-of-fold predictions from folds 1–5 plus 2022 if the single season is too noisy.
- **CV folds use recent seasons** because they resemble the test distribution more than, say, the 1970s.
- **Alternatives to check at Step 13:** validation = 2021–2022 (44 races, more stable), or a training window starting in 1981/1994/2006 (E9).

Decision rule: keep the recommended split unless Step 13 shows a validation season with fewer than 18 races or systematic data gaps. Any change is recorded in the report with the reason.

---

## 28. Evaluation Metrics

### 28.1 Row-level classification (winner, podium, DNF)

Accuracy (reported but never alone), Precision, Recall, F1, **ROC-AUC**, **PR-AUC** (primary under imbalance), confusion matrix at the validation-chosen threshold, **log-loss** and **Brier score** (probability quality), and a **calibration curve** (reliability diagram).

### 28.2 Row-level regression (finishing position)

MAE (primary: "off by X places on average"), MSE, RMSE, **R²**, and Spearman ρ.

### 28.3 Race-level metrics (the most meaningful for F1)

| Metric | Definition |
|---|---|
| Winner top-1 accuracy | % of races where argmax win-probability driver actually won |
| Winner top-3 hit rate | % of races where the actual winner is among the top 3 by win probability |
| Podium overlap | mean over races of \|predicted podium ∩ actual podium\| / 3 |
| Exact podium | % of races with all 3 podium drivers in the right order (strict; expected low) |
| Mean Spearman ρ | Rank correlation between predicted and actual finishing order per race |
| Winner log-loss | −mean log p̂(actual winner) using per-race normalised probabilities |

### 28.4 Uncertainty and statistical comparison

- **Bootstrap 95% CI.** Resample **races** (not rows) with replacement 1,000 times from the test set; recompute every metric; report the 2.5th–97.5th percentiles. With about 70 test races, the CI on winner accuracy will be wide (roughly ±10–12 percentage points), and the report must say so.
- **Paired bootstrap difference.** For model A vs model B, compute the metric difference on the same resamples. If the 95% CI of the difference excludes 0, the difference is meaningful. This backs the model-comparison claims (RQ3).
- **Calibration.** Report Brier score before and after calibration.

---

## 29. Experiment Plan

All experiments are evaluated on **CV folds + the 2022 validation season**. Only E10 touches the test set.

| ID | Experiment | Question answered | Setup | Primary metrics | Tier |
|---|---|---|---|---|---|
| E0 | Heuristic baselines | How good is "grid decides everything"? | Grid-order regressor; pole = winner; grid ≤ 3 = podium | Winner top-1, podium overlap, MAE | MUST |
| E1 | Baseline classification | Does a simple model on minimal features beat E0? | LogReg on grid + quali only | PR-AUC, winner top-1 | MUST |
| E2 | Classification with engineered features | Do form, constructor and circuit features add value? (RQ1) | LogReg, DT on full v1 features | PR-AUC, ROC-AUC, race metrics | MUST |
| E3 | Ensemble classification | Do bagging/boosting beat linear/tree? (RQ3) | RF, HGB (XGB) tuned | Same + calibration | MUST |
| E4 | Dedicated classifier vs regression-derived podium | Is one model enough? | Compare podium overlap | Podium overlap | SHOULD |
| E5 | Regression ladder | Best position model; beats grid heuristic? | Dummy → Linear → Ridge/Lasso → DT → RF → HGB | MAE, RMSE, R², Spearman | MUST |
| E6 | Weather contribution + honesty ablation | Does weather help, and is the gain only oracle information? (RQ4) | Best model: no weather / pre-start weather / race-window weather. Also results on the wet-race subset | Δ metrics, wet vs dry | SHOULD |
| E7 | Feature importance + group ablation | Which factors matter most? (RQ2) | Permutation importance; drop-one-family | Δ PR-AUC / Δ MAE | MUST |
| E8 | Hyperparameter tuning effect | How much does tuning help; over/underfitting? | Default vs tuned; depth validation curve; learning curve | Δ metrics; train vs val curves | MUST |
| E9 | Training window / drift | Is old data helpful or harmful? (RQ5) | Train start ∈ {1950, 1981, 1994, 2006} | Val metrics | SHOULD |
| E10 | **Final test evaluation** | Final unbiased performance | Frozen best models refit on ≤2022 → test 2023–2025, bootstrap CI | All §28 metrics + CI | MUST |
| E11 | From-scratch logistic regression | Does our GD implementation match sklearn? | Podium task; compare coefficients and metrics; loss curve | Coefficient diff; PR-AUC | SHOULD |
| E12 | PCA for linear models | Does decorrelating rolling features help LogReg/Linear? | PCA(k ≤ 10) in pipeline | Δ metrics | COULD |
| E13 | DNF prediction | Is DNF predictable pre-race? | LogReg vs HGB | ROC-AUC, PR-AUC vs base rate | COULD |
| E14 | 2026 live drift check | How badly do 2025-trained models degrade under the 2026 rules? | Predict completed 2026 races (never trained on) | Race metrics | COULD |

---

## 30. Final Prediction Pipeline

### 30.1 Inputs for a race to predict

`season`, `round`. The system looks up the race, circuit, grid and qualifying from Jolpica (historical cache, or a fresh pull for a new race after qualifying). It builds features from all races strictly before that race, fetches weather (archive for past races; the Open-Meteo **forecast** API for a future race, COULD), and applies the frozen pipelines.

### 30.2 Steps

1. Build the feature rows for the race's starters (same code as training, `src/features.py`).
2. Winner model → calibrated pᵢ(win) → normalise so Σ = 1 within the race.
3. Podium model → calibrated pᵢ(podium) → optionally rescale so Σ ≈ 3, capped at 1.
4. Regression model → predicted position → rank within race → integer order.
5. Predicted podium = top 3 by the method chosen in E4, shown in win-probability order.
6. If the true result exists (historical race), print actual vs predicted and the race metrics.

### 30.3 How probabilities are generated and calibrated

- Classifiers output `predict_proba`. Class weighting and boosting distort probabilities, so the chosen model is wrapped in `CalibratedClassifierCV` (method = `sigmoid` for small data; `isotonic` if validation data is sufficient). It is fitted on held-out data (the 2022 season or out-of-fold predictions), **never** on training rows it was fitted on.
- Within-race normalisation turns independent per-driver probabilities into a distribution over "who wins this race". This is a simplification: it ignores the joint structure of finishing orders (Future Work: Plackett–Luce / conditional logit, or Monte-Carlo simulation).
- Calibration quality is shown with a reliability diagram and Brier score (validation and test).

### 30.4 User-facing output (CLI: `python -m src.predict --season 2025 --round 24`)

```
==============================================================
 FORMULA 1 RACE PREDICTION  (model v1.0, trained ≤ 2022)
 2025 Abu Dhabi Grand Prix · Yas Marina · 2025-12-07
 Prediction point: after final starting grid
 Weather (reanalysis, oracle proxy): 26 °C, dry, wind 12 km/h
==============================================================
 Pos  Driver            Team          Grid  Win %  Podium %
 ---  ----------------  ------------  ----  -----  --------
  1   Driver A          Team X          1   38.2     78.5
  2   Driver B          Team Y          2   24.9     66.0
  3   Driver C          Team X          4   11.3     41.2
  4   Driver D          Team Z          3    9.8     37.9
  ...
 20   Driver T          Team Q         20    0.1      0.4
--------------------------------------------------------------
 Predicted podium:  P1 Driver A · P2 Driver B · P3 Driver C
 Model confidence:  winner top-1 accuracy on 2023–25 test = XX%
                    (95% CI XX–XX%)
--------------------------------------------------------------
 [If race already happened]
 Actual podium:     P1 ... · P2 ... · P3 ...
 Podium overlap 2/3 · Spearman ρ = 0.71 · Position MAE = 2.4
==============================================================
 Probabilistic estimate from historical patterns. It cannot
 foresee crashes, failures, safety cars or strategy calls and
 does not guarantee any outcome.
```

(Names and numbers above are placeholders, not results.) Outputs are also saved to `outputs/predictions/{season}_{round}.csv`.

### 30.5 Deployment (course topic, kept minimal)

- Models are saved as `models/{task}_{model}_v1.joblib` (whole pipeline including preprocessing and calibration), plus `models/metadata_v1.json` (feature list, training seasons, metrics, library versions, seed).
- CLI script = deployment artefact. (COULD) Streamlit page with a season/round picker and a win-probability bar chart.

---

## 31. Project Folder Structure

```
F1-Race-Prediction/
├── README.md                    # how to install, run, reproduce
├── requirements.txt             # pinned versions
├── config.yaml                  # seasons, split years, paths, seed, API settings
├── .gitignore                   # data/raw, data/interim, models/*.joblib
├── docs/
│   ├── MASTER_PROJECT_BLUEPRINT.md   # this file
│   ├── DATA_DICTIONARY.md
│   └── DECISIONS_LOG.md         # dated log of every design change + reason
├── data/
│   ├── raw/        (jolpica/, openmeteo/, _manifest.csv)
│   ├── interim/
│   ├── processed/
│   └── external/   (COULD: circuit_meta.csv, constructor_lineage.csv)
├── notebooks/
│   ├── 01_api_exploration.ipynb
│   ├── 02_collect_f1_data.ipynb
│   ├── 03_collect_weather.ipynb
│   ├── 04_validation_cleaning.ipynb
│   ├── 05_integration_master.ipynb
│   ├── 06_eda.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_leakage_audit_and_splits.ipynb
│   ├── 09_classification.ipynb
│   ├── 10_regression.ipynb
│   ├── 11_tuning_ensembles_experiments.ipynb
│   ├── 12_final_evaluation.ipynb        # ONLY notebook that reads test.csv
│   └── 13_final_prediction_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py        # loads config.yaml, sets seeds
│   ├── api_client.py    # cached GET with retries (Jolpica + Open-Meteo)
│   ├── collect.py       # season loops, pagination
│   ├── weather.py       # time alignment + window aggregation
│   ├── cleaning.py      # JSON → flat typed tables, cleaning rules
│   ├── integrate.py     # joins + assertions
│   ├── features.py      # all leakage-safe feature builders
│   ├── splits.py        # temporal split + season-fold CV generator
│   ├── models.py        # pipelines + search spaces
│   ├── logreg_scratch.py# (SHOULD) NumPy logistic regression
│   ├── evaluate.py      # row + race-level metrics, bootstrap
│   ├── plots.py         # consistent figure style
│   └── predict.py       # CLI entry point
├── tests/
│   ├── test_leakage.py  # T1–T6
│   └── test_integrity.py
├── models/
├── outputs/
│   ├── figures/  tables/  predictions/  logs/
└── report/
    ├── report.docx / report.pdf
    ├── slides.pptx
    └── figures/ (copied final figures)
```

Why this is better than notebooks only: logic lives in `src/` (tested, reusable, identical at train and predict time), and notebooks only *call* it and show results. That is what makes the pipeline reproducible and the demo reliable.

---

## 32. Implementation Roadmap

Each step lists: Objective · Inputs · Tasks · Expected output · Files · Depends on · Validation check · Common errors · Done when.

**Step 1: Environment and repository setup**
- Objective: a reproducible workspace.
- Inputs: this blueprint.
- Tasks: create the folder tree (§31); create a venv (Python 3.11+); install pandas, numpy, matplotlib, seaborn, scikit-learn, requests, pyyaml, joblib, jupyter, pytest (xgboost later if needed); pin versions; `git init`; `.gitignore`; `config.yaml` with seed 42 and split years.
- Expected output: `pip freeze` matches `requirements.txt`.
- Files: `requirements.txt`, `config.yaml`, `README.md`, `.gitignore`, `docs/DECISIONS_LOG.md`.
- Depends on: —
- Validation check: `python -c "import sklearn, pandas; print(sklearn.__version__)"` works; a fresh clone installs cleanly.
- Common errors: installing globally instead of in the venv; OneDrive syncing large data folders (consider keeping `data/` excluded from sync).
- Done when: the environment is reproducible from `requirements.txt`.

**Step 2: Jolpica API exploration**
- Objective: confirm access, structure, pagination and limits.
- Inputs: base URL.
- Tasks: request `/2024/1/results.json`; inspect the JSON tree; check the status code; test `limit`/`offset`; read the response headers; read Jolpica's rate-limit docs; record findings.
- Expected output: a short "API facts" cell (a copy of Appendix A, re-verified).
- Files: `notebooks/01_api_exploration.ipynb`.
- Depends on: 1.
- Validation check: one race retrieved with 20 results; `total` field understood.
- Common errors: assuming `limit=1000` works (it caps at 100); forgetting the `.json` suffix.
- Done when: one race can be retrieved and parsed into a DataFrame.

**Step 3: Cached API client**
- Objective: reliable, polite, resumable downloads.
- Inputs: Step 2 findings.
- Tasks: implement `get_json` (cache, retries, back-off, logging, User-Agent) as in §11.3.
- Expected output: a second call for the same URL reads from disk.
- Files: `src/api_client.py`, `data/raw/_manifest.csv`.
- Depends on: 2.
- Validation check: pull the network cable (or use a bad URL) and see a graceful failure log.
- Common errors: caching error responses as if they were valid; not checking `MRData.total`.
- Done when: cache hit/miss and retry behaviour are demonstrated.

**Step 4: Collect schedules and results (1950–2025)**
- Objective: complete raw F1 data. **Checkpoint 2**
- Tasks: loop over seasons; paginate; save raw pages; collect qualifying 1994–2025 and sprint 2021–2025.
- Expected output: about 550 raw JSON files.
- Files: `notebooks/02_collect_f1_data.ipynb`, `src/collect.py`, `data/raw/jolpica/**`.
- Depends on: 3.
- Validation check: per season, the sum of result rows in pages equals `MRData.total`; 76 schedule files; the total is about 25,873 rows.
- Common errors: race `Results` split across pages (merge by season+round); hitting the hourly limit (resume later thanks to caching).
- Done when: the row-count reconciliation table shows 0 differences.

**Step 5: Flatten races into a circuit/date table**
- Objective: the input list for the weather step.
- Tasks: parse schedules into `races.csv` (race_id, date, time, circuit_id, lat, lon, country).
- Files: `data/interim/races.csv`.
- Validation check: about 1,149 rows; no missing lat/lon; `time` null exactly for ≤2004.
- Done when: the table is saved and checked.

**Step 6: Collect and aggregate weather**
- Objective: race weather features. **Checkpoint 3**
- Tasks: one archive request per race (`timezone=auto`); compute the start hour (UTC time → local via `utc_offset_seconds`, or 14:00 local for ≤2004); aggregate the race window [start, start + 2 h] and the pre-start window [start − 3 h, start − 1 h]; sanity-check `wx_is_wet` against about 10 races remembered as wet (e.g. 2021 Belgian, 2011 Canadian, 2008 British, 2016 Brazilian, 2024 São Paulo GP; confirm each from race reports) and set the rain threshold.
- Files: `notebooks/03_collect_weather.ipynb`, `src/weather.py`, `data/raw/openmeteo/*.json`, `data/interim/weather_race.csv`.
- Depends on: 5.
- Validation check: 100% race coverage; plausible ranges (−5 to 45 °C; humidity 0–100); night races (Singapore, Bahrain, Las Vegas) use the correct local evening hours.
- Common errors: mixing local time and UTC; off-by-one hour around daylight saving; treating the precipitation unit (mm) as a probability.
- Done when: `weather_race.csv` is complete and the wet-race sanity table is saved.

**Step 7: Raw data validation**
- Objective: know the data before changing it.
- Tasks: verify the [ASSUMPTION] items: positionText codes, grid = 0 meaning, Indy 500 filter, duplicate keys, position ordering for retirees, status vocabulary per era, missingness per column per decade.
- Files: `notebooks/04_validation_cleaning.ipynb` (part 1), `outputs/tables/validation_report.csv`.
- Validation check: every assumption in this document is marked confirmed or corrected in `DECISIONS_LOG.md`.
- Common errors: "fixing" data before measuring the problem.
- Done when: the validation report is complete.

**Step 8: Cleaning**
- Objective: typed, flat interim tables.
- Tasks: apply the §17 rules; log counts.
- Files: `src/cleaning.py`, `data/interim/{results,qualifying,sprint}.csv`, `outputs/tables/cleaning_log.csv`.
- Validation check: dtypes are correct; no string numbers; exclusion counts logged.
- Done when: the cleaning log reconciles raw row counts to interim row counts.

**Step 9: Integration into the master base**
- Objective: one table, one row per driver-race. **Checkpoint 4**
- Tasks: joins per §18; assertions.
- Files: `notebooks/05_integration_master.ipynb`, `src/integrate.py`, `data/interim/master_base.csv`, `tests/test_integrity.py`.
- Validation check: unique key; one winner per race; qualifying match ≥ 95% (1994+).
- Common errors: a many-to-many join inflating rows; qualifying keyed by car number instead of driver_id.
- Done when: integrity tests pass.

**Step 10: EDA**
- Objective: understand patterns; inform features and models. **Checkpoint 5**
- Tasks: the §22 figures (1–12 and 15–17 now; 13–14 after Step 11).
- Files: `notebooks/06_eda.ipynb`, `outputs/figures/eda_*.png`, `outputs/tables/eda_findings.csv`.
- Validation check: each figure has a one-line takeaway.
- Done when: the findings table has at least 10 "observation → implication" rows.

**Step 11: Feature engineering**
- Objective: the leakage-safe feature set v1. **Checkpoint 6** (with Step 12)
- Tasks: implement feature families (§20) in `src/features.py`; derived standings; validate derived standings against the API standings sample (≥1991 seasons should match exactly, allowing for sprint points).
- Files: `notebooks/07_feature_engineering.ipynb`, `data/processed/master_features.csv`.
- Depends on: 9.
- Validation check: spot-check 3 drivers by hand (e.g. one rookie, one champion, one mid-season team switcher).
- Common errors: forgetting `shift(1)`; rolling across the mixed order of all drivers instead of grouping; computing team features from rows that include the teammate's current result.
- Done when: all features exist and the spot-checks pass.

**Step 12: Leakage audit**
- Objective: prove there is no leakage.
- Tasks: implement T1–T6 (§19.5); fill in the audit table.
- Files: `tests/test_leakage.py`, `notebooks/08_leakage_audit_and_splits.ipynb`, `outputs/tables/leakage_audit.csv`.
- Validation check: `pytest` is green.
- Done when: all tests pass and the audit table is complete.

**Step 13: Temporal split and CV folds**
- Objective: freeze the evaluation design.
- Tasks: inspect races and rows per season 2015–2025; confirm or adjust §27.3; write `train/val/test.csv`; implement the season-fold generator.
- Files: `src/splits.py`, `data/processed/{train,val,test}.csv`, `outputs/tables/split_summary.csv`.
- Validation check: T5 passes; the split summary table (seasons, races, rows, winner count) is saved.
- Done when: the split is recorded in `DECISIONS_LOG.md`.

**Step 14: Feature selection**
- Tasks: §21 steps 1–4 on training/CV data; freeze `feature_list_v1.json`.
- Files: `outputs/tables/feature_selection_log.csv`, `data/processed/feature_list_v1.json`.
- Done when: the list is frozen with a reason for every drop.

**Step 15: Baseline models (E0, E1, dummy + linear)**
- Objective: the bars to beat. **Checkpoint 7** (pre-submission ready)
- Tasks: heuristic, dummy, LogReg (winner, podium), Linear Regression; evaluation functions (`src/evaluate.py`) incl. race-level metrics.
- Files: `notebooks/09_classification.ipynb`, `notebooks/10_regression.ipynb`, `outputs/tables/results_baselines.csv`.
- Validation check: the heuristic metrics look sensible (e.g. the pole-to-win rate matches EDA figure 5 for the same seasons).
- Common errors: evaluating race metrics on rows sorted differently from predictions; reporting accuracy only.
- Done when: the baseline table exists for CV + validation.

**Step 16: Intermediate models (E2, E5 part)**
- Tasks: Decision Tree, Ridge, Lasso, (KNN/SVM optional) with full features; depth validation curve.
- Files: same notebooks; `outputs/figures/model_dt_depth_curve.png`.
- Done when: all intermediate models are in the comparison table.

**Step 17: Ensemble models (E3, E5)**
- Objective: bagging and boosting. **Checkpoint 8**
- Tasks: RF, HGB (XGB COULD) for all targets with default parameters first.
- Files: `notebooks/11_tuning_ensembles_experiments.ipynb`.
- Done when: the default-parameter ensemble results are in the table.

**Step 18: Hyperparameter tuning (E8)**
- Objective: tuned models. **Checkpoint 9**
- Tasks: §26 searches with season-fold CV; save `cv_results_`; learning curve for the best model.
- Files: `outputs/tables/tuning_{model}_{task}.csv`, `models/*_tuned.joblib`.
- Common errors: using `cv=5` (random KFold) by accident; tuning on 2022.
- Done when: best parameters are recorded for each model × task.

**Step 19: Experiments on validation (E4, E6, E7, E9, E11, [E12, E13])**
- Tasks: run and tabulate; choose the final model per task by the pre-declared rule: *best validation primary metric; if within the CI overlap, prefer the simpler model.*
- Files: `outputs/tables/exp_E*.csv`, figures.
- Done when: final models are chosen and written into `DECISIONS_LOG.md`, **before** looking at test data.

**Step 20: Calibration and race-level decoding**
- Tasks: calibrate the chosen classifiers; reliability diagrams; implement the §30 decoding.
- Files: `outputs/figures/calibration_*.png`, `models/final_*_v1.joblib`, `models/metadata_v1.json`.
- Done when: the Brier score improves or the reason it doesn't is understood.

**Step 21: Final test evaluation (E10)**
- Objective: unbiased final numbers. **Checkpoint 10**
- Tasks: refit on ≤2022; evaluate once on 2023–2025; bootstrap CIs; paired bootstrap vs the heuristic; confusion matrices; ROC/PR curves.
- Files: `notebooks/12_final_evaluation.ipynb`, `outputs/tables/final_test_results.csv`.
- Common errors: going back to change models after seeing test results. If that happens, disclose it in the report.
- Done when: the final table with CIs is saved.

**Step 22: Prediction pipeline and demo**
- Objective: user-facing predictions. **Checkpoint 11**
- Tasks: `src/predict.py` CLI; demo notebook; predictions for every 2025 race saved; choose 2–3 showcase races (one easy, one upset, one wet).
- Files: `src/predict.py`, `notebooks/13_final_prediction_demo.ipynb`, `outputs/predictions/*.csv`.
- Validation check: running the CLI on a fresh terminal works in under 10 s using cached data.
- Done when: the demo runs offline end-to-end.

**Step 23: Report, slides and viva preparation**
- Objective: submission package. **Checkpoint 12**
- Tasks: write the report (§38); build the slides (§39); rehearse the demo; practise the viva questions (§40); final reproducibility run from raw cache.
- Done when: every item in the §41 checklist is ticked.

---

## 33. Checkpoints

| # | Checkpoint | Must be completed | Evidence |
|---|---|---|---|
| 1 | API working | Steps 1–3; cached client retrieves one race | Notebook 01 output |
| 2 | Raw F1 data collected | Schedules, results, qualifying, sprint 1950–2025; reconciliation = 0 diff | `_manifest.csv`, reconciliation table |
| 3 | Weather collected | 100% races; wet-race sanity check | `weather_race.csv`, sanity table |
| 4 | Master dataset created | Validation report, cleaning log, `master_base.csv`, integrity tests pass | Tests green |
| 5 | EDA completed | ≥12 figures + findings table | `outputs/figures/eda_*` |
| 6 | Feature engineering completed | `master_features.csv`, leakage tests T1–T6 pass, audit table, frozen split | Tests green |
| 7 | Baselines completed | Heuristic + dummy + LogReg + Linear results on CV/val | `results_baselines.csv` |
| — | **Pre-submission** (date TBC with instructor) | Checkpoints 1–7 + short progress note | Package from CP1–7 |
| 8 | Ensembles completed | DT, RF, HGB (+opt.) results for all targets | Comparison table |
| 9 | Tuning completed | Search results + best params per model/task | `tuning_*.csv` |
| 10 | Final evaluation completed | E10 with bootstrap CIs; final model choice logged before test | `final_test_results.csv` |
| 11 | Prediction pipeline completed | CLI + demo notebook + 2025 predictions | Demo run |
| 12 | Report + presentation + viva prep | §41 checklist complete | Submission package |

---

## 34. Expected Outputs

**Datasets:** `races.csv`, `results.csv`, `qualifying.csv`, `sprint.csv`, `weather_race.csv`, `master_base.csv`, `master_features.csv`, `train/val/test.csv`, `feature_list_v1.json`.

**Tables (`outputs/tables/`):** validation_report, cleaning_log, eda_findings, leakage_audit, split_summary, feature_selection_log, results_baselines, results_classification, results_regression, tuning_*, exp_E4…E13, final_test_results (with CI), feature_importance.

**Figures (`outputs/figures/`):** 17 EDA figures (§22) plus model figures: ROC curves (all classifiers, one plot per task), PR curves, confusion matrices (best model, winner and podium), calibration curves (before/after), predicted vs actual position scatter, residual plot, DT depth validation curve, learning curve, permutation importance bar, feature-group ablation bar, weather ablation bar, E9 training-window line plot, bootstrap CI forest plot comparing models, from-scratch LogReg loss curve, example race prediction bar chart.

**Models:** `models/final_{winner,podium,position}_v1.joblib`, `metadata_v1.json`.

**Predictions:** `outputs/predictions/2025_*.csv` (+ optional 2026 demo).

---

## 35. Limitations

| Limitation | Effect | How it is reported |
|---|---|---|
| **Historical data quality** | Early-era entries (shared drives, DNQs, varying classification rules) are noisier | Validation report; E9 window experiment |
| **Missing data by era** | No qualifying before 1994; no start times before 2005 | Missing flags; imputation documented |
| **Regulation changes** | Points systems, car rules, tyre rules and qualifying formats all changed, so relationships drift | Era-comparable features; recent-season CV; E9 |
| **Different eras** | 1950s DNF rates above 40%, 30+ car fields; the modern era is very different | Percentile features; window experiment |
| **Driver/team changes** | Team renames reset constructor history; mid-season driver swaps | Optional lineage mapping; rolling features react with lag |
| **Circuit changes** | Layout changes under the same circuitId | Circuit history treated as approximate |
| **Weather** | Reanalysis grid is 9–25 km; showers and local rain are poorly captured; air temperature ≠ track temperature; race-window weather is not a forecast | §19.6; E6; labelled "oracle proxy" |
| **Class imbalance** | About 5% winners; accuracy is misleading | PR-AUC, race-level metrics |
| **DNF unpredictability** | Crashes and failures are largely random pre-race | E13 expected near base rate; stated honestly |
| **Unmodelled race events** | Safety cars, strategy, team orders, penalties during the race | Upper bound on achievable accuracy |
| **Temporal drift** | Models trained on the past degrade in new rule eras (2022, 2026) | Temporal test; E14 |
| **Small test set** | About 70 races make CIs wide | Bootstrap CIs always shown |
| **Independence assumption** | Row-level models ignore that exactly one driver wins | Race-level normalisation; Future Work: ranking models |
| **No practice data** | Misses the most direct pre-race pace signal | Future Work |

---

## 36. Risk Management

| Risk | Likelihood | Impact | Mitigation | Trigger / fallback |
|---|---|---|---|---|
| Jolpica unavailable or rate-limited | Medium | High | Cache everything early (Week 1); resumable scripts | If down for more than 2 days, use the Kaggle "Complete Dataset" (Ergast schema) as a documented fallback |
| Open-Meteo limits or outage | Low | Medium | 1 req/s throttle; cache | Proceed without weather; E6 becomes Future Work |
| Hidden leakage inflates results | Medium | **Very high** | §19 rules; tests T1–T6; suspicion test | Any "too good" metric (e.g. winner accuracy far above the heuristic) triggers an audit |
| Scope creep (too many models/sources) | High | High | MUST/SHOULD/COULD tiers; 2-source rule | Drop COULD items first |
| Time overrun beyond 20 h | High | Medium | Checkpoints with weekly targets; baselines first | Cut E12–E14, XGBoost, Streamlit |
| Assumption wrong (e.g. positionText codes) | Medium | Medium | Step 7 validation | Update rules and log in DECISIONS_LOG |
| Models don't beat the grid heuristic | Medium | Medium (for grades) | This is a valid finding; emphasise analysis | Report honestly, with CIs; analyse where ML helps (mid-field, wet races) |
| Notebook/state reproducibility issues | Medium | Medium | Logic in `src/`; "restart & run all" before each checkpoint | Final full rerun at CP12 |
| Data lost (OneDrive sync conflicts) | Low | High | Git for code; raw cache re-creatable | Re-run collection from scripts |
| Team coordination issues (if team) | Medium | Medium | Component ownership + shared `src/` interfaces | Weekly integration run |

---

## 37. Future Work

1. **Practice-session pace** via an additional source (e.g. the FastF1 library / official timing, 2018+). This is the single most justified extra dataset.
2. **Genuine forecast weather** from the Open-Meteo Historical Forecast API (archived forecasts) instead of reanalysis.
3. **Learning-to-rank / Plackett–Luce / conditional logit** models that respect "exactly one winner and a full ordering".
4. **Monte-Carlo race simulation** to produce P(finish = k) for every position.
5. **Pairwise driver-vs-driver models.**
6. **Circuit clustering** (K-Means/DBSCAN on circuit characteristics and processionality), feeding cluster ID as a feature (course unsupervised topic).
7. **Anomaly detection** for data-quality flags and "upset" races.
8. **Neural networks** (MLP) only after tabular baselines, if data grows. Not justified at this size.
9. **Online updating across the 2026 season** to study adaptation to the new regulations.
10. **Streamlit dashboard** with live post-qualifying predictions.

---

## 38. Report Structure

Proposed structure (the requested 14 chapters, with a dedicated leakage/evaluation-design chapter added because it is central to validity):

| Ch. | Title | What to write | Figures | Tables | Results |
|---|---|---|---|---|---|
| — | Front matter | Title, abstract (≤250 words with key numbers + CI), declaration, contents | — | — | Headline test metrics |
| 1 | Introduction | F1 context; why prediction is hard; contributions; report outline | Pipeline diagram (§9) | — | — |
| 2 | Problem Definition | Prediction point T₀; tasks; RQ1–5; hypotheses; scope | Prediction timeline graphic | Targets table (§14) | — |
| 3 | Literature Review | 6–10 sources: sports outcome prediction, F1 prediction studies, ranking models, temporal validation, calibration. Identify the gap | — | Related-work comparison | — |
| 4 | Dataset and Data Collection | Sources, justification, API design, rate limits, caching, Kaggle decision | Coverage-by-era chart | Data sources; endpoints; request budget | Row counts |
| 5 | Data Preprocessing | Validation findings, cleaning rules, integration, population definition | Missingness heatmap | Cleaning log; join checks | Rows kept/dropped |
| 6 | Exploratory Data Analysis | Key patterns and their modelling implications | EDA 2–12, 16–17 | EDA findings table | — |
| 7 | Feature Engineering and Selection | Families, formulas, era normalisation, selection steps | Correlation heatmap; importance | Feature list with categories A/B/C/D | Selection log |
| 8 | Leakage Prevention and Evaluation Design | T₀ rule, forbidden list, audit, tests, temporal split, season CV, metrics definitions | Split timeline diagram | Leakage audit; split summary | Test outcomes (green) |
| 9 | Methodology and Model Development | Each algorithm: intuition, key equation, why chosen, hyperparameters; calibration; decoding | DT depth curve; LogReg-scratch loss curve | Hyperparameter spaces | Best params |
| 10 | Experimental Results | E0–E9, E11–E13 on CV/validation | ROC, PR, calibration, residuals, ablations | Per-experiment tables | All validation metrics |
| 11 | Model Comparison and Final Test | E10: final test with bootstrap CIs; paired comparisons vs heuristic; error analysis (upsets, wet races) | CI forest plot; confusion matrices | Final test table | Answers to RQ1–5 |
| 12 | Final Prediction System | Architecture, CLI, sample outputs, calibration, deployment | Race prediction chart | Example race output | 2–3 showcase races |
| 13 | Limitations | §35, honestly | — | Limitations table | — |
| 14 | Conclusion and Future Work | Answers to the RQs, hypotheses accepted/rejected, lessons, §37 | — | — | Summary numbers |
| — | References, Appendices | API facts, data dictionary, full hyperparameter results, reproducibility instructions, AI-assistance disclosure if required | — | — | — |

---

## 39. Presentation Structure (about 12–15 minutes + demo)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Project, name(s), course |
| 2 | The question | RQ; "does pole decide everything?" hook |
| 3 | Prediction point | T₀ timeline; allowed vs forbidden info |
| 4 | Data pipeline | Jolpica + Open-Meteo → master dataset; row counts |
| 5 | Dataset design | One row = driver × race; targets |
| 6 | Key EDA | Grid-vs-finish heatmap; pole-win by decade; wet vs dry |
| 7 | Features | Families + example of shift/rolling; era normalisation |
| 8 | Leakage audit | Forbidden list; tests T1–T6 passing |
| 9 | Evaluation design | Season-fold CV diagram; 2022 val; 2023–25 test |
| 10 | Models | Ladder: heuristic → linear → tree → RF → boosting |
| 11 | Results: classification | Table + ROC/PR; race-level top-1 / podium overlap with CI |
| 12 | Results: regression | MAE/R² table; predicted vs actual |
| 13 | What matters | Permutation importance; feature-group and weather ablation |
| 14 | **Live demo** | CLI on one easy, one upset, one wet 2025 race |
| 15 | Limitations and future work | Honest list |
| 16 | Conclusion | Answers to RQs; one-sentence takeaway |

Demo script: (1) show the command; (2) explain the columns; (3) compare with the actual result; (4) point out one failure and explain why (e.g. a DNF or rain). Have screenshots ready as a backup if the live run fails.

---

## 40. Viva Questions

Format: **Q** · *Short answer* · Deeper answer (if the examiner follows up).

### A. Project understanding
1. **Q: What exactly does your system predict?** · *Winner, podium and finishing position per driver, before the race starts.* · Three separate supervised models on driver-race rows, aggregated per race into win probabilities, a predicted podium and an order.
2. **Q: When is the prediction made?** · *After the starting grid is final, before the race.* · That point decides which features are legal. Grid and qualifying are allowed; anything from the race is not.
3. **Q: What is your main finding?** · *[Fill in after E10]: e.g. "ensembles beat the grid heuristic on podium overlap by X (CI …)".* · Always quote it with the CI and the comparison baseline.
4. **Q: Why is this a real-world ML problem and not a toy?** · *Noisy, imbalanced, drifting, grouped temporal data with leakage traps.* · Same challenges as demand forecasting or credit scoring: temporal validation, calibration, imbalance.
5. **Q: Why not just predict "pole-sitter wins"?** · *That is our baseline. The question is how much ML adds.* · Pole converts to a win well under 100% of the time; errors cluster in mid-field and wet races, where ML may help.

### B. Dataset
6. **Q: What does one row represent?** · *One driver in one Grand Prix.* · Key (race_id, driver_id); about 25.9k rows 1950–2025 before exclusions.
7. **Q: Why exclude the Indianapolis 500?** · *It was a championship round 1950–60 but in practice a separate series with different drivers.* · Keeping it would distort driver history and circuit stats.
8. **Q: Why not include 2026?** · *The season is incomplete and has new regulations.* · Using it would mix eras; optional E14 uses it only as a drift demonstration.
9. **Q: How many positive samples for "winner"?** · *About one per race (about 1,140 of about 25k rows).* · About 4–5% overall, which motivates PR-AUC and class weights.
10. **Q: How do you handle drivers who did not start?** · *They are excluded from modelling rows, because non-starters are known at T₀.* · They are kept in interim tables so history is complete.

### C. API
11. **Q: Why Jolpica and not Kaggle?** · *Maintained, Ergast-compatible, reproducible and up to date to 2025.* · Kaggle copies are static snapshots of unclear curation. We only keep one as an emergency fallback.
12. **Q: How did you handle pagination?** · *limit=100 (the API maximum), offset loop until `total`.* · Results of one race can span pages, so we merge by (season, round).
13. **Q: Rate limits?** · *Throttled to 1 request/s with exponential back-off, and every response cached.* · About 550 requests in total; caching makes reruns cost zero requests.
14. **Q: How do you ensure reproducibility if the API changes?** · *Raw JSON is saved immutably with a manifest (URL, timestamp).* · The whole pipeline rebuilds from the cache without network access.

### D. Data preprocessing
15. **Q: How did you define DNF?** · *positionText R/N; status text only as a cross-check.* · Status labels changed format around 2023 (coarse "Retired"/"Lapped"), so positionText is more stable.
16. **Q: What about missing qualifying before 1994?** · *NaN plus a missing flag, imputed with grid inside the pipeline.* · Tree models can use the flag; the imputer is fitted on training folds only.
17. **Q: What did grid = 0 mean?** · *Pit-lane start (verified at Step 7).* · Recoded to the back of the grid, with a flag.
18. **Q: Why store raw, interim and processed separately?** · *Traceability and reproducibility.* · Any number in the report can be traced back to the raw bytes.

### E. Feature engineering
19. **Q: Give an example of an engineered feature.** · *Driver's last-5-race average finish.* · Computed per driver, sorted by date, shift(1) then rolling(5), so the current race is excluded.
20. **Q: Why points share instead of points?** · *Points systems changed; share and rank are comparable across eras.*
21. **Q: How do you handle rookies?** · *History features are NaN; count features and a rookie flag tell the model.* · HGB uses NaN natively; others use median imputation with an indicator.
22. **Q: How are constructor features computed without teammate leakage?** · *Aggregate to team-race first, then shift, then roll.* · Otherwise a teammate's current finishing position would leak into the feature.
23. **Q: Why no one-hot driver IDs?** · *They overfit, and don't generalise to new drivers or teams.* · Performance history captures "who is good" in a way that transfers.

### F. Data leakage
24. **Q: What is data leakage?** · *Training with information that would not be available at prediction time.* · It gives optimistic metrics that collapse in real use.
25. **Q: Give three leakage examples in your project.** · *Current race points, standings after the current round, laps completed.* · Also: scaling on the full data, random splits, rolling features without a shift.
26. **Q: How did you verify there is no leakage?** · *Automated tests: recompute-from-past, perturbation, cold-start, suspicion AUC, split disjointness, column denylist.*
27. **Q: Is weather leakage?** · *Race-window reanalysis is technically observed during the race; we label it an oracle-forecast proxy and ablate it.* · A deployable system would need a forecast; the pre-start window is the more honest variant.
28. **Q: Is grid position leakage? It's set on race day.** · *No. It is final before the start, which is our prediction point.*

### G. Classification
29. **Q: Why logistic regression?** · *Interpretable, probabilistic, strong linear baseline.* · Maximum-likelihood fit of a Bernoulli model; coefficients are log-odds effects.
30. **Q: Explain the logistic cost function.** · *Binary cross-entropy (log-loss).* · −[y log p + (1−y) log(1−p)]; convex, so gradient descent reaches the global minimum. Our from-scratch version shows the loss curve.
31. **Q: Why did you (not) use SVM?** · *Optional: RBF-SVM scales about O(n²) and gives no native probabilities.* · A linear SVM with calibration is a fair comparison.
32. **Q: Why not SMOTE?** · *Synthetic driver-race rows break the race structure; class weights plus race-level decoding are cleaner.*
33. **Q: How do you get one winner per race from a binary classifier?** · *Normalise the probabilities within a race and take the argmax.*
34. **Q: What threshold did you use?** · *For row metrics, the F1-optimal threshold on validation; race-level decoding needs no threshold.*

### H. Regression
35. **Q: Why is finishing position regression and not classification?** · *It is ordinal with a natural distance; MAE "places off" is interpretable.* · 20-class classification would ignore that P2 is closer to P1 than P15.
36. **Q: How do you turn continuous predictions into positions?** · *Rank the predictions within each race.*
37. **Q: Ridge vs Lasso?** · *L2 shrinks coefficients (good with collinearity); L1 can zero them (feature selection).*
38. **Q: What does your R² mean?** · *The share of variance in finishing position explained.* · Moderate R² is expected: random events are a large part of the variance.
39. **Q: Why MAE as the primary metric?** · *Robust to outliers (e.g. early DNFs) and easy to interpret.*

### I. Random Forest
40. **Q: How does a Random Forest work?** · *Many decorrelated trees on bootstrap samples with random feature subsets, averaged.* · Lowers variance compared with one deep tree.
41. **Q: Why does it overfit less than a single tree?** · *Averaging reduces variance, and feature subsampling decorrelates the trees.*
42. **Q: What is out-of-bag error?** · *Error on the rows each tree didn't see.* · But OOB is random, not temporal, so it is optimistic here; we rely on season folds.
43. **Q: Impurity vs permutation importance?** · *Impurity is biased toward high-cardinality or continuous features; permutation importance on validation is more reliable.*

### J. Boosting
44. **Q: Bagging vs boosting?** · *Bagging trains in parallel to reduce variance; boosting trains sequentially to reduce bias.*
45. **Q: What does the learning rate do?** · *It shrinks each tree's contribution; smaller rates need more trees and generalise better.*
46. **Q: Why HistGradientBoosting instead of XGBoost?** · *Similar algorithm, built into scikit-learn, handles NaN, fast.* · XGBoost was optional/[used if added].
47. **Q: How did you prevent boosting from overfitting?** · *Early stopping, min_samples_leaf, L2 regularisation, a limited number of leaves.*

### K. Hyperparameter tuning
48. **Q: Grid vs randomized search?** · *Grid for small spaces; randomized search explores large spaces efficiently.*
49. **Q: Which scoring did you tune on?** · *PR-AUC for classifiers, MAE for regression.* · Accuracy would favour "always predict no".
50. **Q: How did you avoid leakage during tuning?** · *The full pipeline is refitted in each season fold; validation and test seasons are never in the search.*
51. **Q: How much did tuning help?** · *[E8 result]* · Compare default vs tuned with CIs; often small, which is a valid finding.

### L. Cross-validation
52. **Q: Why not k-fold CV?** · *It would train on future seasons and split races.*
53. **Q: Describe your CV.** · *Expanding-window season folds validating 2017–2021.*
54. **Q: Why not sklearn's TimeSeriesSplit directly?** · *It splits by row and can cut a race in half; we split by season.*
55. **Q: What is bootstrap and how did you use it?** · *Resampling races with replacement to estimate metric CIs.* · Races, not rows, because rows within a race are dependent.

### M. Evaluation metrics
56. **Q: Why is accuracy misleading here?** · *With about 5% winners, predicting "never wins" is about 95% accurate.*
57. **Q: ROC-AUC vs PR-AUC?** · *ROC can look good under imbalance; PR focuses on the positive class.*
58. **Q: Read your confusion matrix.** · *TP/FP/FN/TN at the chosen threshold; relate precision and recall to it.*
59. **Q: What is calibration and why does it matter?** · *Predicted 30% should win about 30% of the time.* · Class weights distort probabilities; we fix this with Platt/isotonic calibration on held-out data and check with a reliability diagram and Brier score.
60. **Q: How do you know model A is really better than B?** · *Paired bootstrap over races; the CI of the difference excludes zero.*
61. **Q: What is a confidence interval?** · *A range that would contain the true metric in 95% of repeated samples (under the bootstrap approximation).*

### N. Time-based splitting
62. **Q: Why is 2023–2025 the test set?** · *The most recent complete seasons, the same regulation era, and simulates the future.*
63. **Q: Why is 2022 validation?** · *The first year of the current rules; checks robustness to rule change before testing.*
64. **Q: Is old data (1950s) helpful?** · *Tested in E9.* · Old data adds volume but also drift; report which window won.
65. **Q: What is concept drift here?** · *The relationship between features and outcome changes with regulations and team dominance.*

### O. Weather integration
66. **Q: How did you get weather for each race?** · *Circuit lat/long plus race date from Jolpica → Open-Meteo archive hourly → aggregated over the race window.*
67. **Q: How did you align the time?** · *UTC start from Jolpica (2005+), converted to local using the returned offset; 14:00 local assumed before 2005.*
68. **Q: What is reanalysis?** · *A model reconstruction of past weather, blended with observations, on a grid.* · Not a station observation and not a forecast.
69. **Q: Limitations of the weather data?** · *Grid resolution; showers are missed; air temperature is not track temperature; estimated start times before 2005.*
70. **Q: Did weather help?** · *[E6 result]*, including the wet-race subset.

### P. Limitations
71. **Q: Biggest limitation?** · *Unpredictable race events plus a small test set, so wide CIs.*
72. **Q: Would this work for 2026?** · *Probably worse: new rules cause drift.* · E14 (optional) measures it.
73. **Q: What happens with team renames?** · *Constructor history resets; an optional lineage map fixes it.*
74. **Q: Could your model be used for betting?** · *Not designed or validated for that; probabilities are approximate, and we make no guarantee.*

### Q. Future work
75. **Q: What would you add first?** · *Practice pace data and genuine forecast weather.*
76. **Q: Why no neural network?** · *Tabular, about 25k rows: tree ensembles are the strong standard; an ANN adds complexity with little expected gain.* · Could be tried as an MLP comparison later.
77. **Q: How could you model the full race order properly?** · *Ranking models (Plackett–Luce) or Monte-Carlo simulation.*
78. **Q: How would you deploy this for real?** · *Scheduled pull after qualifying → feature build → saved pipeline → web page.* · Plus monitoring of drift and recalibration each season.

---

## 41. Final Deliverables

Checklist for submission (confirm the format with the instructor):

- [ ] **Project report** (PDF), structure §38, with figures and tables, references and appendices
- [ ] **Code repository** (zip or Git link): `src/`, `notebooks/` (run top to bottom), `tests/`, `requirements.txt`, `config.yaml`, `README.md` with reproduction steps
- [ ] **Data:** processed datasets (or instructions plus scripts to rebuild from the API); raw cache optional because of size
- [ ] **Saved models** + `metadata_v1.json`
- [ ] **Outputs:** figures, tables, predictions
- [ ] **Presentation** slides (§39)
- [ ] **Demo:** working CLI (+ backup screenshots/video)
- [ ] **Viva preparation:** this §40 bank, filled with the actual numbers
- [ ] **Pre-submission package** (Checkpoints 1–7)
- [ ] `docs/DECISIONS_LOG.md` showing each design change
- [ ] Plagiarism/originality: cite the data sources (Jolpica/Ergast, Open-Meteo with ERA5 attribution), libraries and literature; disclose AI-tool assistance if the university requires it

---

## 42. Step-by-Step Next Actions, Time Budget and Roles

### 42.1 Hour budget (per the handout's about 20 h)

| Block | Steps | Hours | Tier |
|---|---|---|---|
| Setup + API exploration + client | 1–3 | 2.0 | MUST |
| F1 collection + weather | 4–6 | 2.5 | MUST/SHOULD |
| Validation + cleaning + integration | 7–9 | 3.0 | MUST |
| EDA | 10 | 1.5 | MUST |
| Feature engineering + leakage audit + split + selection | 11–14 | 3.5 | MUST |
| Baselines + intermediate + ensembles | 15–17 | 3.0 | MUST |
| Tuning + experiments + calibration | 18–20 | 2.5 | MUST/SHOULD |
| Final test + prediction CLI | 21–22 | 2.0 | MUST |
| **Technical subtotal** | | **20.0** | |
| Report + slides + rehearsal | 23 | +4–6 | MUST |

The 20 h estimate is realistic only if COULD items are dropped first and the report is written alongside the work (a paragraph per checkpoint), not all at the end.

### 42.2 Calendar (today = 25 Sep 2026; the project is assessed in the last two weeks, about 16–30 Nov)

| Week | Dates | Target |
|---|---|---|
| 1 | 28 Sep – 4 Oct | CP1–3 (APIs, raw data, weather) |
| 2 | 5 – 11 Oct | CP4 (validated master dataset) |
| 3 | 12 – 18 Oct | CP5–6 (EDA, features, leakage tests, split) |
| 4 | 19 – 25 Oct | CP7 (baselines) → **pre-submission ready** |
| 5 | 26 Oct – 1 Nov | CP8–9 (ensembles, tuning, experiments) |
| 6 | 2 – 8 Nov | CP10–11 (final test, prediction CLI) |
| 7 | 9 – 15 Nov | CP12 (report, slides, viva rehearsal) |
| Buffer | 16 Nov → | Fixes; assessment |

### 42.3 Team roles (only if the instructor allows teams)

The handout says each student's solution is assessed separately, so **every member must be able to explain every part**. If teams are allowed:

| Member | Owns | Also contributes |
|---|---|---|
| M1: Data engineering | Steps 2–9 (API client, collection, weather, cleaning, integration, data dictionary) | Report Ch. 4–5 |
| M2: Features and evaluation design | Steps 10–14 (EDA, features, leakage tests, splits, selection) | Report Ch. 6–8 |
| M3: Modelling and system | Steps 15–22 (models, tuning, experiments, calibration, final test, CLI) | Report Ch. 9–12 |
| All | Step 23; weekly integration run; viva rehearsal | Ch. 1–3, 13–14, slides |

For an individual project, follow the steps in order; the checkpoints are the milestones.

---

## Appendix A: API Facts Verified on 2026-09-25

| Fact | Evidence |
|---|---|
| Jolpica base URL live, JSON, no auth | `GET /ergast/f1/2024/1/results.json` → HTTP 200, Cloudflare, `Cache-Control: max-age=600` |
| Max page size 100 | `limit=1000` returned `limit: "100"` |
| Results fields | number, position, positionText, points, Driver{driverId, code, dateOfBirth, nationality}, Constructor{constructorId, name}, grid, laps, status, Time{millis}, FastestLap |
| Race `time` present 2005+, null ≤2004 | 1990 and 2004 null; 2005, 2006 present (UTC "Z") |
| Qualifying from 1994; Q1 only ≤2004; Q1+Q2 in 2005; Q1–Q3 in 2006 | 1993 → 0 rows; samples 1994, 2003, 2005, 2006 |
| Schedule includes session dates, not practice results | 2024 R1 keys: FirstPractice, SecondPractice, ThirdPractice, Qualifying |
| Standings per round are *after* that round | `/2024/5/driverStandings` returns round 5 cumulative (110 pts leader) |
| No constructor standings in 1957 | total = 0 |
| Status vocabulary changed | 2010 granular (Engine, +1 Lap, Collision…); 2023/2024 coarse (Finished, Lapped, Retired, Did not start, Disqualified); 136 values all-time |
| Pit stops from 2011 | 2010 R1 = 0, 2011 R1 = 45 |
| Sprint endpoint from 2021 | 2021 total = 60 |
| Indianapolis 500 in data | 1950 includes "Indianapolis 500" |
| Totals | 1,172 races listed (incl. 23 scheduled 2026 rounds); 26,181 results (incl. 308 from 2026) |
| 2025 complete | 24 races, 479 results |
| 2026 incomplete | 23 scheduled, 308 results so far |
| Open-Meteo archive works for 1950 and 2024 | Silverstone 1950-05-13 daily + hourly returned; Sakhir 2024-03-02 15:00 UTC: 18.2 °C, 41% RH, 0 mm, cloud 99%, wind 11.5 km/h, WMO 3 |
| Open-Meteo snaps to grid | Requested 26.0325, 50.5106 → returned 26.0457, 50.5142, elevation 8 m |

## Appendix B: Questions to Ask the Instructor

1. Individual or team project? If team, what size, and how is individual contribution assessed?
2. Pre-submission date and expected contents?
3. Report format, page limit and citation style?
4. Is using public APIs (Jolpica, Open-Meteo) acceptable, and must raw data be submitted?
5. Is a from-scratch implementation of any algorithm expected in the project, as the course overview suggests?
6. Rubric weights for report / code / presentation / demo / viva?

---

## WHAT WE SHOULD DO FIRST

1. **Ask the instructor the Appendix B questions**, especially individual vs team and the pre-submission date, because they change the plan.
2. **Set up the repository and environment (Step 1):** folder tree, venv, pinned `requirements.txt`, `config.yaml` (seasons 1950–2025, seed 42, split years), `git init`, and an empty `DECISIONS_LOG.md`.
3. **Re-run the API exploration in `01_api_exploration.ipynb` (Step 2):** retrieve 2024 Round 1 results and qualifying and one Open-Meteo race-day response, and check Jolpica's documented rate limit.
4. **Build the cached, retrying API client (Step 3)** and prove that a second identical request is served from disk.
5. **Collect one pilot season end-to-end (2024: schedule, results, qualifying, sprint, weather)** into `data/raw/`, and reconcile the row counts against `MRData.total`, before launching the full 1950–2025 collection.
