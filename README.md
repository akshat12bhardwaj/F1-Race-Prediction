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
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

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
| 1 | Environment and repository setup | In progress |
| 2 | Jolpica API exploration | Not started |

## Reproducibility

- Random seed 42 everywhere (`config.yaml`).
- Package versions pinned in `requirements.txt`.
- Raw API responses are cached in `data/raw/` and every later file is rebuilt from them by code.
- Every design change is recorded in `docs/DECISIONS_LOG.md`.
