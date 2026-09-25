# Data Dictionary

This file describes every dataset file and column in the project. A table is added or updated **when the file is actually created**, using the real column names and types. The planned schema is in blueprint §13. This dictionary records what really exists.

Category codes (blueprint §13): **A** = directly from source · **B** = derived by feature engineering · **C** = optional · **D** = needs an additional data source.

---

## File inventory

| File | Stage | Created at step | Rows (approx.) | Status |
|---|---|---|---|---|
| `data/raw/jolpica/**.json` | Raw | 4 | about 550 files | Not created |
| `data/raw/openmeteo/*.json` | Raw | 6 | about 1,149 files | Not created |
| `data/raw/_manifest.csv` | Raw log | 3 | one per request | Not created |
| `data/interim/races.csv` | Interim | 5 | about 1,149 | Not created |
| `data/interim/results.csv` | Interim | 8 | about 25,873 | Not created |
| `data/interim/qualifying.csv` | Interim | 8 | — | Not created |
| `data/interim/sprint.csv` | Interim | 8 | — | Not created |
| `data/interim/weather_race.csv` | Interim | 6 | about 1,149 | Not created |
| `data/interim/master_base.csv` | Interim | 9 | about 25,873 | Not created |
| `data/processed/master_features.csv` | Processed | 11 | — | Not created |
| `data/processed/train.csv`, `val.csv`, `test.csv` | Processed | 13 | — | Not created |
| `data/processed/feature_list_v1.json` | Processed | 14 | — | Not created |

---

## Column template

Copy this table for each file when it is created:

### `<file name>`
One row = `<what one row represents>` · Primary key: `<columns>`

| Column | Type | Cat. | Source / formula | Pre-race? | Missing values | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## Tables

*(No data files have been created yet. The first entries come at Step 3/4.)*
