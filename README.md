# ML-Based Formula 1 Race Outcome Prediction

CSE20232 Machine Learning (AY 2026–27), end-semester project.

This project predicts Formula 1 Grand Prix outcomes (finishing position, winner, podium) using only information available **before the race starts**. It compares linear, tree-based and ensemble models against a simple "grid order" baseline.

The full design is in [docs/MASTER_PROJECT_BLUEPRINT.md](docs/MASTER_PROJECT_BLUEPRINT.md).

## Data sources

- [Jolpica F1 API](https://api.jolpi.ca/ergast/f1/) (Ergast-compatible): races, results, qualifying, sprint
- [Open-Meteo Historical Weather API](https://open-meteo.com/): hourly reanalysis weather (ERA5-based)

## Setup (Windows, PowerShell)

Requires **Python 3.11**.

```powershell
# 1. Create the virtual environment OUTSIDE OneDrive (avoids syncing ~500 MB of packages)
py -3.11 -m venv "$env:USERPROFILE\.venvs\f1-race-prediction"

# 2. Activate it
& "$env:USERPROFILE\.venvs\f1-race-prediction\Scripts\Activate.ps1"

# 3. Install the pinned dependencies (run from the project folder)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Verify the environment
python -m pytest tests/test_environment.py -v

# 5. Register the environment as a Jupyter kernel (once per machine)
python -m ipykernel install --user --name f1-race-prediction --display-name "Python 3.11 (f1-race-prediction)"
```

In VS Code, open a notebook and choose the kernel **Python 3.11 (f1-race-prediction)**.

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Using the cached API client (`src/api_client.py`)

```python
import logging
from src.api_client import JolpicaClient

logging.basicConfig(level=logging.INFO)   # optional: show cache hits/misses and requests

client = JolpicaClient()                  # settings come from config.yaml
races = client.get_race_results(2024, 1)  # list of races (here 1) with all records merged
quali = client.get_qualifying(2024, 1)
sprint = client.get_sprint_results(2024, 5)
season = client.get_race_results(2024)    # whole season: paginates automatically (limit=100)
schedule = client.get_schedule(2024)

page = client.get_json("/2024/5/driverStandings.json", {"limit": 100})  # any other endpoint
```

- Returned data is the API's JSON, **unchanged**: all values are strings. Type conversion happens in the cleaning step.
- `get_race_results`, `get_qualifying` and `get_sprint_results` return `[]` when the API has no data (e.g. no sprint that weekend).
- Temporary failures (HTTP 429, 5xx, timeouts, connection errors) are retried with waits of 2, 4, 8, 16 and 32 s. HTTP 400/404 raise `JolpicaAPIError` immediately.

**Cache location:** `data/raw/jolpica/`. The file path mirrors the request URL, with parameters sorted, for example:
`/2024/1/results.json?limit=100&offset=0` → `data/raw/jolpica/2024/1/results__limit-100__offset-0.json`.
Each download is logged in `data/raw/_manifest.csv`; permanent failures go to `data/raw/_failed_requests.csv`.

**Bypass the cache** for one request: `client.get_race_results(2024, 1, refresh=True)` (downloads again and overwrites the file).
**Clear the cache:** delete the folder `data/raw/jolpica/` (or one file inside it). The next request downloads it again.

## Historical collection 1950–2025 (Step 5)

```powershell
python -m src.collect --start 1950 --end 2025
# small test run into separate files:
python -m src.collect --start 1950 --end 1951 --output-prefix subset_
```

- **Runtime:** first run ≈ 100 minutes (≈ 800 requests, 7.5 s apart to stay under Jolpica's 500 requests/hour). Afterwards everything comes from the cache and a full re-run takes about 1 minute with **0** requests.
- **Resume:** if the run stops (network, laptop sleep, Ctrl+C), just run the same command again. Every response already downloaded is read from `data/raw/jolpica/`; only missing ones are downloaded.
- **Outputs:**
  - `data/interim/races.csv`, `results.csv`, `qualifying.csv`, `sprints.csv`, `drivers.csv`, `constructors.csv` (raw API text values, all seasons)
  - `outputs/tables/historical_collection_summary.csv`: one row per season, compared with the API's `MRData.total`
  - `outputs/tables/historical_race_validation.csv`: one row per race, with counts and flags
  - `outputs/tables/historical_edge_cases.csv`: unusual observations (one row each)
  - `outputs/tables/historical_status_vocabulary.csv` and `historical_field_coverage.csv`
  - `outputs/logs/collection_*.log` (per run) and `outputs/logs/collection_runs.csv` (request counts, runtime)
- **Validate:** `python -m pytest tests/ -v`. The season summary's `matches_api_totals` column must be `True` for every season.
- Nothing is removed or recoded at this stage; duplicates and odd records are **flagged** in the reports.

## Historical data policies (Steps 5.5–5.6)

```powershell
python -m src.policy_analysis   # read-only analysis of historical edge cases (Step 5.5)
python -m src.policies          # apply the APPROVED policies P-01…P-11 (Step 5.6)
```

- The approved policies (shared drives, shared positions, qualifying gaps, grid = 0, status categories, DNF definition, ambiguous rows, model population, constructor identity, order exceptions, Indianapolis 500) are described with rationale and consequences in `docs/DECISIONS_LOG.md`.
- They are applied as **extra columns** on copies of the interim tables: `data/interim/results_policy.csv` and `races_policy.csv`. The original collected files are never changed or reduced.
- `in_model_population = True` marks the 25,066 rows that the later model dataset will use (drivers who started, primary entry, not the Indianapolis 500).
- Counts: `outputs/tables/policy_application_summary.csv`.

## Historical race weather (Step 6)

```powershell
python -m src.weather                                   # all races 1950-2025
python -m src.weather --start 2023 --end 2024 --output-prefix subset_   # small test run
```

- **Source:** Open-Meteo Historical Weather API (archive, ERA5-based **reanalysis**, CC-BY 4.0). These are reconstructed historical conditions, **not** a forecast that was available before the race.
- **Runtime:** first run ≈ 20 minutes (≈ 1,215 requests at 1 per second; free limits 600/min, 5,000/h). Re-runs use the cache (`data/raw/openmeteo/`) and make 0 requests. An interrupted run is resumed by running the same command.
- **Start times:** official UTC start for 2007–2025. For **2005–2006** the API's "UTC" values are really local clock times, so they are read as local time and converted (P-12, approved). **1950–2004 has no official start time, so 14:00 local is assumed.** All three cases are recorded in `start_time_source`.
- **Local times** use Python's timezone database (historical daylight-saving rules). Open-Meteo's own local-time option applies one fixed offset to all dates, so it is not used (D-013).
- **Indianapolis 500 (P-11):** not requested and never model-usable; the 2000–2007 US GPs at the same circuit are included.
- **Outputs:** `data/interim/weather_race.csv` (one row per race, prestart and race-window summaries), `data/interim/weather_hourly.csv` (hourly values −6…+6 h around the start), and reports `outputs/tables/weather_*.csv`. Log: `outputs/logs/weather_*.log`.

## Data validation (Step 7)

```powershell
python -m src.validate_raw     # read-only; about 30 s; no API requests
```

- Validates races, results, qualifying, sprints, drivers, constructors, the approved policy columns (P-01…P-11, recomputed from raw data) and the weather data (P-12, recomputed start times and window summaries), plus 14 cross-table relationships.
- Outputs in `outputs/tables/`: `raw_validation_summary.csv` (PASS/FAIL/INFO per check), `raw_missingness.csv` (every missing value classified), `raw_cross_table_validation.csv`, `raw_anomalies.csv` (`known_handled` vs `review`).
- Nothing is changed: no rows removed, no values imputed or recoded, no targets or features. The run stops with an error if any input file changes.
- **Limitation:** 1950–2004 weather windows use an **assumed 14:00 local** start, and historical reanalysis is **not** a pre-race forecast.

## Exploratory data analysis (Step 8)

```powershell
python -m src.eda              # read-only, deterministic; about 30 s
```

Open `notebooks/03_eda.ipynb` (kernel **Python 3.11 (f1-race-prediction)**) for the narrative. Outputs: `outputs/tables/eda_*.csv` (31) and `outputs/figures/eda/*.png` (19). The EDA is descriptive only: no targets, features, splits or imputation. Outcome variables are labelled POST-RACE in `eda_variable_roles.csv`. Weather is reanalysis, not a forecast (1950–2004 assumed 14:00 local start).

## Running tests

```powershell
python -m pytest tests/ -v                  # everything (one test needs internet)
python -m pytest tests/ -v -m "not live"    # offline only
python -m pytest tests/test_api_client.py -v
python tests/smoke_api_client.py            # live smoke test: at most 3 real requests
```

## Project structure

```
F1-Race-Prediction/
├── config.yaml        settings: seasons, split years, paths, API settings, seed
├── requirements.txt   pinned Python packages
├── docs/              blueprint, decisions log, data dictionary
├── data/raw/          exact API responses (never edited, not in git)
├── data/interim/      cleaned flat tables (not in git)
├── data/processed/    model-ready datasets
├── data/external/     optional manual tables
├── notebooks/         step-by-step notebooks (01 … 13)
├── src/               reusable Python code used by the notebooks
├── tests/             automated checks (environment, integrity, leakage)
├── models/            saved trained pipelines
├── outputs/           figures, tables, predictions, logs
└── report/            report, slides and final figures
```

## Progress

| Step | Description | Status |
|---|---|---|
| 1 | Environment and repository setup | Complete |
| 2 | Jolpica API exploration (`notebooks/01_api_exploration.ipynb`) | Complete |
| 3 | Cached API client (`src/api_client.py`) | Complete |
| 4 | 2024 pilot collection (`notebooks/02_pilot_collection_2024.ipynb`, `src/collect.py`) | Complete |
| 5 | Historical collection 1950–2025 (`python -m src.collect`, `src/collection_checks.py`) | Complete: 1,149 races, 25,873 results |
| 5.5 | Historical data policy analysis (`python -m src.policy_analysis`) | Complete |
| 5.6 | Apply approved policies P-01…P-11 (`python -m src.policies`) | Complete, **policies APPROVED** |
| 6 | Historical race weather (`python -m src.weather`) | Complete: 1,138/1,138 eligible races |
| 6.1 | P-12: 2005–2006 start times read as local clock (APPROVED) | Complete: 1,138 races model-usable, 0 quality errors |
| 7 | Raw data validation (`python -m src.validate_raw`) | Complete: 193 PASS, 7 INFO, 0 FAIL |
| 8 | Exploratory data analysis (`python -m src.eda`, `notebooks/03_eda.ipynb`) | Complete, awaiting review: 31 tables, 19 figures |

## Reproducibility

- Random seed 42 everywhere (`config.yaml`).
- Package versions pinned in `requirements.txt`.
- Raw API responses are cached in `data/raw/` and every later file is rebuilt from them by code.
- Every design change is recorded in `docs/DECISIONS_LOG.md`.
