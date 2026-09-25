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
| 5 | Historical collection 1950–2025 (`python -m src.collect`, `src/collection_checks.py`) | Complete, awaiting review: 1,149 races, 25,873 results |
| 6 | Weather collection | Not started |

## Reproducibility

- Random seed 42 everywhere (`config.yaml`).
- Package versions pinned in `requirements.txt`.
- Raw API responses are cached in `data/raw/` and every later file is rebuilt from them by code.
- Every design change is recorded in `docs/DECISIONS_LOG.md`.
