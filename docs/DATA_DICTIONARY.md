# Data Dictionary

This file describes every dataset file and column in the project. A table is added or updated **when the file is actually created**, using the real column names and types. The planned schema is in blueprint §13. This dictionary records what really exists.

Category codes (blueprint §13): **A** = directly from source · **B** = derived by feature engineering · **C** = optional · **D** = needs an additional data source.

---

## File inventory

| File | Stage | Created at step | Rows (approx.) | Status |
|---|---|---|---|---|
| `data/raw/jolpica/**.json` | Raw | 3–4 | about 550 files | **Created** (2024 pilot + Step 3 smoke test: 18 files) |
| `data/raw/openmeteo/*.json` | Raw | 6 | about 1,149 files | Not created |
| `data/raw/_manifest.csv` | Raw log | 3 | one per request | **Created** (18 rows) |
| `data/interim/2024_races.csv` | Interim (pilot) | 4 | 24 | **Created** |
| `data/interim/2024_results.csv` | Interim (pilot) | 4 | 479 | **Created** |
| `data/interim/2024_qualifying.csv` | Interim (pilot) | 4 | 479 | **Created** |
| `data/interim/2024_sprints.csv` | Interim (pilot) | 4 | 120 | **Created** |
| `data/interim/2024_drivers.csv` | Interim (pilot) | 4 | 25 | **Created** |
| `data/interim/2024_constructors.csv` | Interim (pilot) | 4 | 10 | **Created** |
| `outputs/tables/pilot_2024_race_summary.csv` | Output table | 4 | 24 | **Created** |
| `outputs/tables/pilot_2024_status_combinations.csv` | Output table | 4 | 10 | **Created** |
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

### Conventions for the Step 4 pilot tables (`data/interim/{season}_*.csv`)

- Created by `src/collect.py` (`collect_season`) from cached Jolpica responses. Notebook: `notebooks/02_pilot_collection_2024.ipynb`.
- **Every value is the API's raw text**, not converted or cleaned. Numbers such as `grid` or `points` are stored as text in the API and written as they are. Read with `pd.read_csv(path, dtype=str)` to keep them as text.
- **Empty cell = the API did not send this field** (e.g. Q3 for a driver knocked out earlier, `time_*` / `fastest_lap_*` for retirements). Nothing is invented.
- The "Pre-race?" column says whether the value is known before the race starts (blueprint §19). **No = outcome, never a model input for that race.**

### `data/raw/_manifest.csv`
One row = one successful network download (cache hits are not logged).

| Column | Meaning |
|---|---|
| fetched_at_utc | Download time (UTC, ISO 8601) |
| path | API path, e.g. `/2024/results.json` |
| params | Query parameters as sorted JSON, e.g. `{"limit": 100, "offset": 0}` |
| status | HTTP status (always 200; failures go to `_failed_requests.csv`) |
| n_bytes | Size of the saved response |
| mrdata_total | `MRData.total` of the response |
| cache_file | Project-relative path of the cached raw file |

### `2024_races.csv`
One row = one race in the season calendar · Primary key: `race_id` (= season + round) · Source: `/{season}/races.json`

| Column | Type (stored) | Cat. | Source field | Pre-race? | Missing | Notes |
|---|---|---|---|---|---|---|
| season | text | A | `season` | Yes | 0 | |
| round | text | A | `round` | Yes | 0 | "1"–"24" |
| race_id | text | B | built: `{season}_{round:02d}` | Yes | 0 | e.g. `2024_01` (blueprint §13.1) |
| race_name | text | A | `raceName` | Yes | 0 | |
| race_date | text | A | `date` | Yes | 0 | YYYY-MM-DD |
| race_time | text | A | `time` | Yes | 0 in 2024 | UTC, e.g. `15:00:00Z`; absent before 2005 (Appendix A) |
| circuit_id | text | A | `Circuit.circuitId` | Yes | 0 | |
| circuit_name | text | A | `Circuit.circuitName` | Yes | 0 | |
| circuit_lat, circuit_lon | text | A | `Circuit.Location.lat/long` | Yes | 0 | For the weather lookup (Step 6) |
| locality, country | text | A | `Circuit.Location.locality/country` | Yes | 0 | |
| sprint_scheduled | True/False | B | built: calendar entry has a `Sprint` session | Yes | 0 | 6 of 24 in 2024 |

### `2024_results.csv` and `2024_sprints.csv` (same columns)
One row = one driver in one race (or sprint) · Key: (`season`, `round`, `driver_id`), 0 duplicates in 2024 · Source: `/{season}/results.json`, `/{season}/sprint.json` (paginated)

| Column | Type (stored) | Cat. | Source field | Pre-race? | Missing (race / sprint 2024) | Notes |
|---|---|---|---|---|---|---|
| season, round, race_id, race_name, race_date, race_time, circuit_id, circuit_name, locality, country | text | A/B | race object | Yes | 0 | Copied from the race the record belongs to |
| driver_id | text | A | `Driver.driverId` | Yes | 0 | Stable across seasons (F-001) |
| driver_code | text | A | `Driver.code` | Yes | 0 | e.g. VER |
| permanent_number | text | A | `Driver.permanentNumber` | — | 0 | Driver's **current** number, **not** historical (F-001). Do not use as a feature |
| driver_number | text | A | `number` (result) | Yes | 0 | Car number used in **this** race |
| driver_first_name, driver_last_name | text | A | `Driver.givenName/familyName` | Yes | 0 | |
| driver_dob | text | A | `Driver.dateOfBirth` | Yes | 0 | For driver age later |
| driver_nationality | text | A | `Driver.nationality` | Yes | 0 | |
| constructor_id | text | A | `Constructor.constructorId` | Yes | 0 | Renamed teams get new IDs (F-001) |
| constructor_name, constructor_nationality | text | A | `Constructor.name/nationality` | Yes | 0 | |
| grid | text | A | `grid` | **Yes** | 0 / 0 | "0" once in 2024 (OCO, R20); non-starters also have a grid value |
| position | text | A | `position` | **No** | 0 / 0 | Numeric for every row, including DNF/DNS/DSQ |
| position_text | text | A | `positionText` | **No** | 0 / 0 | Number, or R / W / D in 2024 |
| points | text | A | `points` | **No** | 0 / 0 | |
| laps | text | A | `laps` | **No** | 0 / 0 | |
| status | text | A | `status` | **No** | 0 / 0 | 2024: Finished, Lapped, Retired, Did not start, Disqualified |
| time_millis, time_text | text | A | `Time.millis`, `Time.time` | **No** | 47 / 4 | Absent for non-finishers |
| fastest_lap_rank, fastest_lap_lap, fastest_lap_time | text | A | `FastestLap.rank/lap/Time.time` | **No** | 17 / 4 | |
| fastest_lap_avg_speed, fastest_lap_speed_units | text | A | `FastestLap.AverageSpeed.speed/units` | **No** | 17 / 4 | Units "kph" |

### `2024_qualifying.csv`
One row = one driver in one qualifying session · Key: (`season`, `round`, `driver_id`), 0 duplicates · Source: `/{season}/qualifying.json` (paginated)

| Column | Type (stored) | Cat. | Source field | Pre-race? | Missing (2024) | Notes |
|---|---|---|---|---|---|---|
| race and driver/constructor columns | text | A/B | as in results | Yes | 0 | Same names as results |
| qualifying_position | text | A | `position` | **Yes** | 0 | |
| q1 | text | A | `Q1` | **Yes** | 0 | Lap time text, e.g. `1:30.031` |
| q2 | text | A | `Q2` | **Yes** | 119 | Absent for drivers knocked out in Q1 (≈5 per race) |
| q3 | text | A | `Q3` | **Yes** | 239 | Absent for drivers knocked out in Q1/Q2 (≈10 per race) |

### `2024_drivers.csv`
One row = one driver in the season's driver list · Key: `driver_id` · Source: `/{season}/drivers.json`
Columns: season, driver_id, driver_code, permanent_number, driver_first_name, driver_last_name, driver_dob, driver_nationality (all text, category A). **Note:** the list can include drivers with no race record (2024: `antonelli`).

### `2024_constructors.csv`
One row = one constructor in the season list · Key: `constructor_id` · Source: `/{season}/constructors.json`
Columns: season, constructor_id, constructor_name, constructor_nationality (all text, category A).

### `outputs/tables/pilot_2024_race_summary.csv`
One row per race: season, round, race_id, race_name, number_of_results, number_of_qualifying_records, number_of_sprint_records, duplicate_result_keys, missing_driver_ids, missing_constructor_ids (counts).

### `outputs/tables/pilot_2024_status_combinations.csv`
Counts of observed (source, status, position_text_type) combinations. position_text_type = "numeric" or the letter code. Input for the Step 7 DNF/DSQ/non-starter decision.
