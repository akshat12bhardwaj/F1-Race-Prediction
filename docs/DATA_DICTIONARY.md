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
| `data/interim/races.csv` | Interim (raw text) | 5 (historical collection) | 1,149 | **Created** 1950–2025 |
| `data/interim/results.csv` | Interim (raw text) | 5 | 25,873 | **Created** |
| `data/interim/qualifying.csv` | Interim (raw text) | 5 | 10,973 | **Created** (1994–2025 only, gaps 1996–2002) |
| `data/interim/sprints.csv` | Interim (raw text) | 5 | 480 | **Created** (2021–2025) |
| `data/interim/drivers.csv` | Interim (raw text) | 5 | 3,248 (season × driver) | **Created** |
| `data/interim/constructors.csv` | Interim (raw text) | 5 | 1,121 (season × constructor) | **Created** |
| `outputs/tables/historical_collection_summary.csv` | Report | 5 | 76 | **Created** |
| `outputs/tables/historical_race_validation.csv` | Report | 5 | 1,149 | **Created** |
| `outputs/tables/historical_edge_cases.csv` | Report | 5 | 3,017 | **Created** |
| `outputs/tables/historical_status_vocabulary.csv` | Report | 5 | 245 | **Created** |
| `outputs/tables/historical_field_coverage.csv` | Report | 5 | per season × endpoint × field | **Created** |
| `outputs/tables/subset_*.csv`, `data/interim/subset_*.csv` | Resume-test evidence | 5 | 1950–1951 | **Created** |
| `data/interim/results_policy.csv` | Interim + approved policy flags | 5.6 | 25,873 | **Created** |
| `data/interim/races_policy.csv` | Interim + approved policy flags | 5.6 | 1,149 | **Created** |
| `outputs/tables/policy_application_summary.csv` | Report | 5.6 | 27 | **Created** |
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

### Historical tables 1950–2025 (`data/interim/races.csv`, `results.csv`, `qualifying.csv`, `sprints.csv`, `drivers.csv`, `constructors.csv`)

**The columns are identical to the 2024 pilot tables described below** (same code: `src/collect.py`), with all seasons stacked in season order and API order within a season. `drivers.csv` / `constructors.csv` have one row per **season × driver/constructor** (the season lists). Same conventions: raw API text, and an empty cell means the API did not send the field.

Observed coverage across 1950–2025 (from `historical_field_coverage.csv` and `historical_collection_summary.csv`):

| Column(s) | Coverage 1950–2025 |
|---|---|
| race_time | **Empty for every race 1950–2004** (731 races); present for every race 2005–2025 |
| fastest_lap_* | Only from **2004** onward |
| time_millis / time_text | All seasons, finishers only |
| driver_code | Present in 35 seasons, partly (e.g. 1.4% of records in some early seasons) |
| permanent_number | Present in 26 seasons, partly; it is the driver's current number anyway (F-001) |
| qualifying (whole table) | **None before 1994.** 1994–2002 only for some races (e.g. 2001: 1 of 17). Every race from 2003 |
| sprints | 2021–2025 only (480 rows) |
| position_text codes | Numbers, **R** (9,121), **W** (358), **D** (159). E, F and N never occur |
| status | 136 distinct values in race results |

Keys: `(season, round, driver_id)` is **not unique** in 1950–1964 (172 rows in 40 races: shared drives). `(season, round, position)` is not unique in 46 races up to 1964 (shared cars). Both are kept and reported; see `historical_edge_cases.csv`.

### `data/interim/results_policy.csv` (Step 5.6, approved policies P-01…P-11)
**All columns of `results.csv`, byte-for-byte the same values**, followed by these derived columns (written by `src/policies.py`; True/False flags unless stated). One row per result row (25,873), nothing removed.

| Column | Policy | Meaning |
|---|---|---|
| entry_count | P-01 | Number of car entries of this driver in this race (1–3) |
| multi_entry_driver | P-01 | entry_count > 1 |
| is_primary_entry | P-01 | The driver's best-placed started entry in this race (tie-break: car number) |
| shared_position | P-02 | Another driver has the same position in this race (co-drivers) |
| shared_position_count | P-02 | Number of drivers credited with this position (integer) |
| race_has_shared_win | P-02 | The race has a shared position 1 |
| race_has_qualifying | P-03 | The race has any qualifying data |
| has_qualifying_row | P-03 | This driver has a qualifying row for this race |
| grid_status | P-04 | given / pit_lane_start / unknown / non_starter (raw `grid` unchanged) |
| pit_lane_start | P-04 | grid_status == pit_lane_start |
| result_category_detail | P-05 | finished, classified_not_running, started_not_classified, disqualified, did_not_start, ambiguous_withdrawn_after_start, ambiguous_classified_withdrawn, ambiguous_classified_not_classified, unknown |
| result_category | P-05/P-07 | result_category_detail with ambiguous rows mapped per P-07 |
| ambiguous_status | P-07 | result_category_detail is one of the ambiguous categories |
| started | P-07/P-08 | result_category ∈ {finished, classified_not_running, started_not_classified, disqualified} |
| retired_while_classified | P-06 | result_category == classified_not_running (NOT a DNF) |
| laps_order_exception | P-10 | Row is part of a laps inversion among non-classified rows |
| is_indianapolis_500 | P-11 | race_name == "Indianapolis 500" |
| in_model_population | P-08 | started ∧ is_primary_entry ∧ ¬is_indianapolis_500 (25,066 rows) |

DNF (P-06) is `result_category == "started_not_classified"` (function `is_dnf`). No DNF target column exists yet.

### `data/interim/races_policy.csv` (Step 5.6)
All columns of `races.csv` plus: result_rows, model_population_rows, race_has_qualifying, race_has_shared_win, race_has_shared_position, race_has_laps_order_exception, is_indianapolis_500, **exclude_from_winner_metrics** (P-02: the 3 shared-win races).

### `outputs/tables/historical_collection_summary.csv`
One row per season: races, races_missing_race_time, result_rows, qualifying_rows, sprint_rows, unique_drivers, unique_constructors (in results), drivers_in_season_list, constructors_in_season_list, duplicate_result_keys / duplicate_qualifying_keys / duplicate_sprint_keys (rows involved), missing_race_ids / missing_driver_ids / missing_constructor_ids, **api_total_*** (`MRData.total` of each endpoint), **matches_api_totals** (True when every count equals the API total).

### `outputs/tables/historical_race_validation.csv`
One row per race: season, round, race_id, race_name, race_date, race_time, sprint_scheduled, result_count, qualifying_count, sprint_count, duplicate_keys, missing_ids, grid_zero, non_numeric_position_text, repeated_position_rows, season_median_result_count, and **flags** (`;`-separated). Flags: no_results, result_count_differs_from_season_median, no_qualifying_in_qualifying_season, qualifying_count_differs_from_results, sprint_scheduled_but_no_sprint_results, sprint_results_but_not_scheduled, missing_race_time, duplicate_keys, missing_ids, grid_zero, repeated_position, indianapolis_500.

### `outputs/tables/historical_edge_cases.csv`
One row per unusual observation: category, season, round, race_id, driver_id, constructor_id, detail. Nothing was removed from the data. Categories:

| Category | Level | Meaning |
|---|---|---|
| result_count_differs_from_season_median | race | Number of result rows differs from the season's median |
| indianapolis_500 | race | Indianapolis 500 round (1950–1960) |
| race_without_qualifying | race | Season has qualifying data but this race has none |
| qualifying_count_differs_from_results | race | Qualifying and result row counts differ |
| season_without_qualifying | season | No qualifying data in the whole season |
| missing_race_time | season | Races without a start time |
| position_text_code | season | Count of each non-numeric positionText code |
| classified_but_not_finished | season | Numeric positionText with a status other than Finished/Lapped/+N Laps |
| grid_zero | record | grid = "0" |
| non_starter | record | Status Did not start / Withdrew / Did not qualify / Did not prequalify, or positionText W/F |
| disqualified_or_excluded | record | Status contains "Disqualified"/"Excluded", or positionText D/E |
| duplicate_result_key, duplicate_qualifying_key, duplicate_sprint_key | record | Rows sharing (season, round, driver_id) |
| repeated_position | record | Rows sharing a position within one race |
| absent_between_appearances | record | Driver raced in rounds r−1 and r+1 of a season but has no row in round r |
| in_qualifying_not_in_results / in_results_not_in_qualifying | record | Driver sets differ between the two sources for one race |
| season_list_driver_without_records | record | In the season driver list, no race/qualifying/sprint row |
| result_driver_not_in_season_list | record | In results, not in the season driver list |
| constructor_id_with_multiple_names, constructor_name_with_multiple_ids, driver_id_with_multiple_identities, same_person_multiple_driver_ids | entity | Identity changes (none found) |
| unexpected_field | season | JSON field not seen in Step 2 (none found) |

### `outputs/tables/historical_status_vocabulary.csv`
Every (source, status, position_text_type) combination: records, first_season, last_season. position_text_type = "numeric" or the letter code. Input for the Step 7 DNF/DSQ/non-starter decision.

### `outputs/tables/historical_field_coverage.csv`
For each endpoint (results, qualifying, sprint), season and level (record, Driver, Constructor): records, records_with_field, share, known_field (seen in Step 2).

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
