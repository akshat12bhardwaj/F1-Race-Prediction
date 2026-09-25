# Decisions Log

Every design decision or change to the blueprint is recorded here with its date and reason. Newest entries go at the bottom.

Format: **ID · Date · Step · Decision · Reason · Blueprint section affected**

---

### D-001 · 2026-09-25 · Step 0
- **Decision:** Adopt `MASTER_PROJECT_BLUEPRINT.md` v1.0 as the project specification.
- **Reason:** Reviewed and finalised before implementation.
- **Affects:** All sections.

### D-002 · 2026-09-25 · Step 1
- **Decision:** Use Python **3.11**.
- **Reason:** The blueprint requires 3.11+. 3.11 is installed on the development machine, and every pinned package has a ready-made Windows wheel for it (checked with a test install).
- **Affects:** §32 Step 1.

### D-003 · 2026-09-25 · Step 1
- **Decision:** Create the virtual environment **outside OneDrive** (`%USERPROFILE%\.venvs\f1-race-prediction`), not inside the project folder.
- **Reason:** The project folder is synced by OneDrive. A venv holds thousands of files (about 500 MB) that would sync constantly and can break. The blueprint lists this as a common Step 1 error.
- **Affects:** §32 Step 1 (common errors).

### D-004 · 2026-09-25 · Step 1
- **Decision:** `requirements.txt` pins the **directly used** packages only. The blueprint's "pip freeze matches requirements.txt" check is interpreted as "the pinned packages appear in `pip freeze` with exactly these versions". A full `requirements-lock.txt` can be generated with `pip freeze` if needed.
- **Reason:** Pinning about 100 indirect dependencies by hand is error-prone. pip resolves them consistently from the direct pins.
- **Affects:** §32 Step 1 (expected output).

### D-005 · 2026-09-25 · Step 2
- **Decision:** Register the project venv as its own Jupyter kernel, **"Python 3.11 (f1-race-prediction)"** (`python -m ipykernel install --user --name f1-race-prediction`). All project notebooks use this kernel, and each notebook's setup cell checks package versions against `requirements.txt`.
- **Reason:** The default `python3` kernel on this machine points to a different Python installation, so notebooks silently ran with the wrong packages. Found during the first Step 2 run.
- **Affects:** Reproducibility (§32 Step 1–2).

### F-001 · 2026-09-25 · Step 2 · Verified API findings (`notebooks/01_api_exploration.ipynb`, 2024 data)
- **Pagination:** default `limit` = 30, max 100 (larger values silently capped). `offset` is exact and contiguous. `MRData.total` = number of result records (not races) and is constant across pages. A full 2024 walk gave 479 = total with 0 duplicates. **Races are split across pages** (2024 rounds 6, 11, 16, 21).
- **Types:** every value, including numbers, is a JSON **string**.
- **Silent "no data":** a nonexistent round or season returns **HTTP 200 with total = 0**. Invalid `limit` values fall back to 30 and `offset=-1` to 0. A non-numeric round gives HTTP 404 (HTML); an unknown endpoint gives HTTP 400 (JSON).
- **Fields:** sprint records have the same fields as race results. Qualifying has number, position, Driver, Constructor, Q1–Q3 (Q2/Q3 only for drivers who reached them). `Time` and `FastestLap` are missing for some non-finishers.
- **Status/position:** `position` is numeric for every driver. Non-finishers are ordered by laps completed (24/24 races). positionText codes seen: R (Retired), **W (status "Did not start", with a non-zero grid value)**, D (Disqualified). **6 "Retired" drivers were classified with a numeric positionText. One disqualification (Belgian GP, RUS) has a numeric positionText "20".**
- **grid = 0:** one 2024 case (OCO, Mexico City GP; qualified 19th). Consistent with a pit-lane start; the API does not state the reason.
- **IDs:** driverIds are stable (21 drivers identical 2023→2024; `hamilton` covers 394 results since 2007). Constructor renames produce **new IDs** (`alfa`→`sauber`, `alphatauri`→`rb`, 2023→2024). The `Driver` object reflects **current** details (e.g. permanentNumber); the result's `number` is the car number at race time.
- **Headers:** no rate-limit or retry headers. `Cache-Control: max-age=600`, Cloudflare. Documented limits: 4 req/s burst, 500 req/h, HTTP 429 when exceeded, "will decrease" in future.
- **Affects:** Step 3 client (check `total`, handle non-JSON errors, 429). The DNF, DSQ and non-starter rules need a decision at Step 7 (see below). **No blueprint rule has been changed.**

### D-006 · 2026-09-25 · Step 3 · **needs approval**
- **Decision:** The cache file path **mirrors the request URL**, with parameters sorted by name: `/{path}.json?limit=100&offset=0` → `data/raw/jolpica/{path}__limit-100__offset-0.json` (e.g. `data/raw/jolpica/2024/results__limit-100__offset-0.json`). The blueprint §15 example was `data/raw/jolpica/results/2024_0000.json`.
- **Reason:** The cache key must depend on the complete request (path + all parameters). One rule then works for every endpoint (results, qualifying, sprint, standings, schedule) and stays deterministic and human-readable. The folder root (`data/raw/jolpica/`) and the manifest are unchanged.
- **Affects:** §15 file patterns, §16 storage layout (naming only).

### D-007 · 2026-09-25 · Step 3
- **Decision:** API client rules:
  (a) `limit` outside 1..100 or `offset` < 0 raises `ValueError` before any request, because the API would silently change these values (F-001).
  (b) Retry only on HTTP 429/500/502/503/504, connection errors and timeouts: at most 5 retries (6 attempts), waits of 2, 4, 8, 16, 32 s from `config.yaml`. HTTP 400/404 and other errors are raised at once.
  (c) Only valid JSON responses containing `MRData` are cached. Errors are never cached. A `total = 0` response **is** cached (it is a valid "no data" answer).
  (d) Cache files hold the exact response bytes, written atomically (temp file + rename). Unreadable cache files are downloaded again.
  (e) Pages are merged by (season, round). Exact duplicates are detected with the key (season, round, position, driverId); position is included so 1950s drivers with two cars in one race are not dropped. The number of unique records must equal `MRData.total`, otherwise `JolpicaAPIError` is raised.
  (f) The client never converts data types.
- **Reason:** Implements the verified Step 2 behaviour (F-001) with the simplest safe rules.
- **Affects:** §11.3 client design.

### D-008 · 2026-09-25 · Step 3
- **Decision:** Test layout: `pytest.ini` (sets `pythonpath = .` so tests can `import src`, and registers a `live` marker). All API-client tests are mocked except one `live` test. The live smoke test is the script `tests/smoke_api_client.py` (not collected by pytest). `src/config.py` currently only loads `config.yaml`; seed-setting will be added when a later step needs it.
- **Reason:** Tests must run without depending on the live API. The smoke test must be a deliberate, manual run.
- **Affects:** §31 folder structure (adds `pytest.ini`).

### D-006 approval · 2026-09-25
- D-006 (URL-mirrored cache naming) **approved** by the project owner after the Step 3 review. The cache naming system stays as implemented.

### D-009 · 2026-09-25 · Step 4
- **Decision:** Step 4 runs as a **2024 pilot** (blueprint "What we should do first", action 5) before the 1950–2025 collection. Details:
  (a) Collection uses **season-level paginated endpoints** (`/{season}/results.json`, `/qualifying.json`, `/sprint.json`, `limit=100`), as blueprint §15 specifies, not one request per race. Pages are split back into races and reconciled race by race.
  (b) Collection and flattening code lives in `src/collect.py` (blueprint §31) so the full collection can reuse it. Flattening keeps the **API's raw text values**; type conversion stays in Step 8.
  (c) Pilot tables are saved as `data/interim/{season}_{table}.csv` (races, results, qualifying, sprints, drivers, constructors). Pilot check tables are saved as `outputs/tables/pilot_{season}_*.csv`.
  (d) Two extra small season lists are collected (`/{season}/drivers.json`, `/{season}/constructors.json`, 1 request each) to reconcile driver and constructor information.
  (e) **Official standings are not collected in the pilot.** Blueprint §15 and §20 use them only to validate our derived standings at Step 11, never as features.
  (f) Primary key checked: `(season, round, driver_id)`. Duplicates are counted and reported, never dropped.
- **Reason:** Proves the pipeline end-to-end on one season with a small request budget (15 requests) before scaling.
- **Affects:** §15 (pilot first), §31 (`src/collect.py`, notebook name `02_pilot_collection_2024.ipynb`).

### F-002 · 2026-09-25 · Step 4 · Verified 2024 pilot findings (`notebooks/02_pilot_collection_2024.ipynb`)
- 24 races (rounds 1–24, no duplicates), `race_time` present for all. Sprint rounds 5, 6, 11, 19, 21, 23; calendar and sprint data agree.
- Row counts equal `MRData.total`: results 479, qualifying 479, sprint 120, drivers 25, constructors 10. Season-level data is identical to the per-round Step 3 responses (R1 results, R1 qualifying, R5 sprint).
- 0 duplicate `(season, round, driver_id)` keys; 0 missing race, driver or constructor IDs. The qualifying driver set equals the results driver set in every race.
- **A non-participating driver has no row at all:** Australian GP (R3) has 19 results. `sargeant` is absent and Williams has one entry. Such drivers are not recorded as "W" rows.
- **The season driver list includes a driver with no records:** `antonelli` is in `/2024/drivers.json` but has no race, qualifying or sprint row. The reason is not in the API data.
- grid = 0 once (OCO, R20). Status × positionText (race): Finished/num 287, Lapped/num 138, Retired/R 43, Retired/num 6, Did not start/W 3, Disqualified/D 1, Disqualified/num 1. Sprint: Finished/num 116, Retired/R 3, Retired/num 1.
- Re-running with a new client: 0 network requests, 15 cache hits, byte-identical CSVs. 0 existing raw files changed.
- **No DNF/DSQ/non-starter rule was applied** (still pending for Step 7).

### D-009 approval · 2026-09-25
- D-009 (2024 pilot approach) **approved** by the project owner after the Step 4 review.

### D-010 · 2026-09-25 · Step 5
- **Decision:** `api.jolpica.min_seconds_between_requests` raised from **1.0 to 7.5 s** in `config.yaml`.
- **Reason:** The full collection needs about 800 requests. At 1 s spacing that is up to 3,600/hour, far above Jolpica's documented **500/hour** sustained limit (F-001), so HTTP 429 would be triggered. 7.5 s = 480/hour. Cache hits are not affected.
- **Affects:** §11.1 (rate limits). Cost: about 95 minutes for the first full collection.

### D-011 · 2026-09-25 · Step 5
- **Decision:** Historical collection runs as a command-line script, `python -m src.collect --start 1950 --end 2025` (also `--output-prefix` for test runs), not a notebook. Reports are built by a new module, `src/collection_checks.py`. Outputs use the blueprint §16 names `data/interim/{races,results,qualifying,sprints,drivers,constructors}.csv`. The 2024 pilot files (`2024_*.csv`) are kept unchanged.
- **Reason:** A 95-minute run is safer as a resumable script with a log file (`outputs/logs/`) than as a notebook cell. The report logic is kept separate from collection so it can be tested offline.
- **Affects:** §31 (adds `src/collection_checks.py`; blueprint notebook `02_collect_f1_data` is replaced by the script).

### F-003 · 2026-09-25 · Step 5 · Verified historical collection findings (1950–2025)
- **Completeness:** 76 seasons, **1,149 races, 25,873 results, 10,973 qualifying, 480 sprint rows**, 3,248 season-driver and 1,121 season-constructor rows. **Every season matches every `MRData.total`.** 817 unique drivers, 203 unique constructors, 77 circuits. 0 missing race, driver or constructor IDs.
- **Run:** 749 new downloads, 485 cache hits (1950–1951 subset + 2024 pilot + total reads), **3 retries, all recovered** (one read timeout, two HTTP 500 on `/2004/qualifying.json`), runtime 94.6 min. Re-running from the cache: **0 requests**, 2.7 s, 17 output files byte-identical, 779 raw files and 780 manifest lines unchanged.
- **Duplicate keys:** `(season, round, driver_id)` repeats in **172 rows / 40 races, all 1950–1964** (shared drives: one driver credited in two cars). Positions repeat in 46 races up to 1964 (shared cars, e.g. 1951 French GP: fangio and fagioli both P1 in car 8). **Kept and reported, not dropped.** Qualifying and sprint keys are unique.
- **Qualifying coverage:** none before 1994. **1994–2002 incomplete** (races with qualifying: 1994 15/16, 1995 17/17, 1996 7/16, 1997 10/17, 1998 7/16, 1999 3/16, 2000 4/17, 2001 1/17, 2002 2/17). Complete for every race 2003–2025.
- **Race start time:** missing for **all** races 1950–2004 (731), present for all 2005–2025.
- **FastestLap** only from 2004.
- **positionText codes in the API: only R (9,121), W (358), D (159).** E, F and N never occur. Drivers who failed to qualify have **no result row**.
- **grid = 0: 267 rows**, mostly non-starters (202 with positionText W, 219 with 0 laps, most often status "Withdrew"). Only 40 are classified finishers.
- **Non-starters** carry many statuses (Withdrew 246, Accident 29, Engine 18, Did not start 9, Injury 8, ...). **Disqualified/Excluded:** Disqualified 153, Excluded 8, Underweight 1.
- **136 distinct race status values**, 245 (source, status, code) combinations.
- **Indianapolis 500:** 11 races (1950–1960). **circuitId `indianapolis` is also used for the United States GP 2000–2007**, so the blueprint's filter by race name is required.
- **Identity:** no constructorId has more than one name, no name has more than one ID, and no driverId has more than one identity (the API supplies one current name per ID). No unexpected JSON fields.
- **Other flags:** 793 absent-between-appearances (mostly before 1990), 142 season-list drivers without records, 363 races whose result count differs from the season median.

### F-004 · 2026-09-25 · Step 5.5 · Historical policy analysis (read-only)
- Produced by `python -m src.policy_analysis` from `data/interim/*.csv`, which was **not modified** (hashes verified before and after). Outputs: `outputs/tables/historical_policy_analysis.csv` (27 rows), `historical_policy_examples.csv` (104), `historical_status_category_proposal.csv` (246), `historical_qualifying_coverage.csv` (36).
- **Shared drives:** 83 duplicate (season, round, driver_id) groups = 172 rows in 40 races (1950–1964). **Every group has a different car number per row**, and `(season, round, driver_id, driver_number)` is unique across all 25,873 rows. Shared positions: 105 groups, 228 rows, 46 races, **all the same car and constructor** (co-drivers). 3 shared wins (1951 French, 1956 Argentine, 1957 British GP).
- **Status:** 136 race statuses. **78 statuses occur in more than one category** (e.g. Engine: R 1,855 / W 18 / numeric 138), so status alone cannot define DNF. Proposed categories (count): finished 15,554 · classified_not_running 674 · started_not_classified 9,121 · disqualified 162 · did_not_start 338 · ambiguous 24 (20 W with laps > 0, all "Withdrew"; 3 classified "Withdrew"; 1 classified "Not classified"). Total 25,873.
- **grid = 0 (267):** 202 non-starters/ambiguous (143 of them 1950–1969); **48 started with laps > 0, all 2015–2024** (pit-lane starts); 17 started categories with 0 laps (13 disqualified).
- **Populations:** 25,511 started rows (candidate model rows before Indy exclusion); 338 non-starters with a row (136 of them with grid > 0); 7 failed-to-qualify rows (visible only 1994+); 142 list-only driver-seasons (62 drivers never raced).
- **Constructors:** 203 IDs ↔ 203 names (strict 1:1). 20 IDs reused after gaps of 5+ seasons (aston_martin 1959–60 & 2021–25, mercedes 1954–55 & 2010–25, alfa 4 blocks, renault 3 blocks...). 28 automatic "succession" candidates mix real renames with driver moves.
- **Retiree ordering:** 7,998 adjacent R pairs, **8 laps inversions** (8 races, 1951–1992, none after 2009). 4 races have an R row above a classified row (1954 Indy 500; 2005 Australian GP: button/sato R at positions 11 and 14; 2023_18 and 2024_14: numeric-code disqualifications).
- **Indianapolis:** `race_name == "Indianapolis 500"` selects exactly 11 races / 405 rows / 107 drivers (4 also raced other GPs: ascari, ruby, ruttman, ward). circuit_id `indianapolis` is also used by the 8 US GPs 2000–2007.

## Historical data policies P-01 … P-11: **APPROVED** (2026-09-25, Step 5.6)

Proposed in Step 5.5 (F-004), **approved by the project owner** and applied in Step 5.6 by `python -m src.policies` (code: `src/policies.py`, tests: `tests/test_policies.py`).

**How they are applied:** as extra columns on copies of the interim tables:
- `data/interim/results_policy.csv`: all 25,873 rows, every original column unchanged, plus 18 policy columns.
- `data/interim/races_policy.csv`: all 1,149 races, plus race-level flags.
- Counts: `outputs/tables/policy_application_summary.csv`.

`results.csv`, `races.csv`, `qualifying.csv` and all raw API files are **not modified**; 813 files were verified by SHA-256 before and after. No target columns, features or recoded grid numbers are created yet.

| ID | Approved policy | Implemented as | Rationale | Consequence (numbers from the applied run) |
|---|---|---|---|---|
| P-01 | Keep every car entry. In the model dataset, a driver with several entries in a race is represented by the **best-placed started entry**; an entry count is kept. Raw entries are never deleted. | `entry_count`, `multi_entry_driver`, `is_primary_entry` (ranking: started first, then lowest position, then car number) | 1950–1964 shared drives are separate car entries with different car numbers; `(season, round, driver_id, driver_number)` is unique | 172 rows of multi-entry drivers; 89 secondary entries kept but outside the model population. Derived season points must later sum **all** entries |
| P-02 | Keep co-driver rows at their official position with a shared-position flag. Shared-win races are excluded **only** from winner-specific metrics. | `shared_position`, `shared_position_count`, `race_has_shared_win`; race level `exclude_from_winner_metrics` | Co-drivers of one car are credited with the same position | 228 rows flagged; 3 races (1951 French, 1956 Argentine, 1957 British GP) excluded from winner metrics only |
| P-03 | Missing qualifying stays missing; never drop a race for it. Availability indicators at race and driver level. | `race_has_qualifying`, `has_qualifying_row` (availability flags only; no values imputed) | No qualifying before 1994, gaps 1994–2002 | 631 races without qualifying kept; 14,907 result rows without a qualifying row kept |
| P-04 | grid = 0: started with laps > 0 → pit-lane start; started with 0 laps → unknown; non-starter → non_starter. Raw grid preserved. No era cut-off. | `grid_status` ∈ {given, pit_lane_start, unknown, non_starter}, `pit_lane_start` | grid 0 is mostly non-starters historically; every started grid-0 row with laps > 0 is 2015–2024 | 267 grid-0 rows unchanged; 48 pit-lane starts; 17 unknown. The numeric pit-lane value (field_size + 1) is set when the model dataset is built |
| P-05 | Preserve raw status and positionText; derive categories separately. | `result_category_detail` (9 categories), `result_category` (after P-07) | Status alone is ambiguous (78 of 136 statuses occur in more than one category) | 136 raw statuses kept. result_category: finished 15,554 · started_not_classified 9,141 · classified_not_running 678 · did_not_start 338 · disqualified 162 |
| P-06 | Primary DNF definition = **started, not classified**. Classified-but-not-running is flagged separately and is not DNF. | `is_dnf(category)` function (no target column yet); `retired_while_classified` | Matches blueprint §14 (positionText R); classified retirements hold a legitimate classified position | 9,141 rows would be DNF; 678 flagged as retired while classified |
| P-07 | The 24 ambiguous rows count as started, with an ambiguity flag. | `ambiguous_status`; mapping: W + laps > 0 → started_not_classified (20); classified "Withdrew"/"Not classified" → classified_not_running (4) | They completed laps or hold a classified position | 24 rows added to the started population, all flagged |
| P-08 | Model population = drivers who actually started, excluding the Indianapolis 500. Non-starters are not counted as starts in history features. | `started`, `in_model_population` (= started ∧ primary entry ∧ not Indy 500) | The prediction point is after the grid is final; non-starters are known then | 25,535 started rows; 338 non-starters excluded; **25,066 model-population rows**, unique per (season, round, driver_id) |
| P-09 | Use API `constructor_id` as-is; no automatic merging by name or rename detection. | No new column; constructor_id untouched | Automatic succession detection mixes real renames with driver moves | 203 IDs unchanged; constructor history restarts when the API ID changes (e.g. alphatauri → rb) |
| P-10 | Use the official API position order; flag the 8 ordering exceptions; never re-sort. | `laps_order_exception` (both rows of each inverted pair); race level `race_has_laps_order_exception` | Order follows laps in 7,990 of 7,998 adjacent pairs | 16 rows / 8 races flagged; positions unchanged |
| P-11 | Exclude races with `race_name == "Indianapolis 500"` (exact); never by circuit_id. | `is_indianapolis_500` (row and race level); used in `in_model_population` | circuit_id `indianapolis` is also used by the 2000–2007 US GPs | 11 races / 405 rows outside the model population; the 8 US GPs stay. History features must also skip these rows |

### D-012 · 2026-09-25 · Step 6 · Weather collection design
- **Decision:**
  (a) Source: Open-Meteo Historical Weather (archive) API, hourly: temperature_2m, relative_humidity_2m, precipitation, rain, cloud_cover, wind_speed_10m, wind_direction_10m, weather_code. No soil/surface variables (no clear link to race conditions).
  (b) New module `src/weather.py` with its own small cached client (same behaviour as the Jolpica client: deterministic cache, 1 s spacing, retries on 429/5xx/timeouts). Cache: `data/raw/openmeteo/{weather,timezones}/{name}__{hash}.json`, where the hash is SHA-1 of the sorted request parameters. Separate logs `data/raw/openmeteo/_manifest.csv` and `_failed_requests.csv`, so no existing Jolpica cache or manifest file is touched.
  (c) One request per race: 72 UTC hours (start date −1 … +1).
  (d) Start time: official UTC for 2005+; **1950–2004 assumed 14:00 local** (no official times, F-003), flagged `imputed_local_14h`.
  (e) Windows relative to H0 (start rounded down): prestart −3…−1 h, race 0…2 h. Accumulated variables use the following-hour values (a+1…b).
  (f) Outputs keep raw observations (`weather_hourly.csv`, −6…+6 h, unchanged values) separate from simple window summaries (`weather_race.csv`: mean/min/max/sum, circular mean for wind direction). **No wet flag, no thresholds** (deferred).
  (g) **Indianapolis 500 (P-11): not requested**, weather_status `excluded_indianapolis_500`, never model-usable; the US GP 2000–2007 is included.
  (h) Sanity checks only flag (severity info/warning/error); `model_use_allowed` = no error-level flag.
- **Reason:** Blueprint §11.2, §13.4, §19.6. The design keeps raw and derived data separate and makes collection resumable and reproducible.
- **Affects:** §11.2 (single `_manifest.csv` → separate Open-Meteo manifest), §13.4.

### D-013 · 2026-09-25 · Step 6 · Weather requested in UTC; local time from zoneinfo
- **Decision:** Request weather with `timezone=GMT` (UTC timestamps). Use Open-Meteo `timezone=auto` **only** to look up each circuit's IANA timezone name (77 cached requests). Compute local times with Python `zoneinfo` (historical daylight-saving rules).
- **Reason (verified):** Open-Meteo's `auto` mode applies **one fixed, current UTC offset to every date**. Las Vegas on 18 Nov 2023 returned `utc_offset_seconds = -25200` (UTC−7), but the correct offset is UTC−8 (DST ended 5 Nov). Its "local" hours matched UTC hours shifted by exactly 7 (65/65), not 8. Trusting it would misalign about half of all races by one hour.
- **Affects:** config `api.openmeteo.timezone` changed from "auto" to "GMT".

### F-005 · 2026-09-25 · Step 6 · Verified weather collection results
- **Requests:** 1,145 network requests + 70 cache hits (the 2023–2024 subset) on the first full run, **0 retries, 0 failures**, 19.4 min. Re-runs: 0 requests, 1,215 cache hits, 9 output files byte-identical. 1,215 raw files (1,138 races + 77 circuit timezones). 800 pre-existing Jolpica raw/manifest/interim files verified unchanged.
- **Coverage:** 1,149 races; 11 Indianapolis 500 excluded; **1,138 / 1,138 eligible races have weather (100%)** in every era. 0 missing hourly values (14,794 hourly rows, 8 variables).
- **Sanity:** no implausible temperature, humidity, precipitation, wind, cloud or weather-code values; no duplicate or irregular timestamps; no missing window hours; all units as expected. Weather grid cell 0.7–8.9 km from the circuit (median 4.2). Race-window ranges: temperature 6.6–36.9 °C, precipitation 0–16.9 mm (321 races > 0 mm, 157 ≥ 0.5 mm), max wind 1.1–41.8 km/h.
- **Start times:** 418 official (2005–2025), 731 imputed (1950–2004). **2005–2006 official `race_time` values are local clock times labelled as UTC** (see P-12). Almost all are "14:00:00Z" across Melbourne, Tokyo, Europe and Montréal, which as UTC gives local starts of 01:00, 23:00, 16:00 and 09:00. The local starts are spread out (R = 0.545 and 0.530) while every season 2007–2025 has R ≥ 0.833. **All 37 races flagged (error), `model_use_allowed = False`, values not changed.**
- Local date differs from the API date in 5 races: Las Vegas 2023–2025 (correct: the API date is the UTC date) and 2005/2006 Australia (a symptom of the clock issue).
- **Wet-race plausibility:** all 9 reference wet races (general-knowledge list, not verified) show precipitation in the reanalysis. Example: 2024 São Paulo, 3.2 mm in the race window.
- **Model-usable weather:** 1,101 races (all 329 races 2010–2025).

## P-12 · 2005–2006 start times: **APPROVED** (2026-09-25, Step 6.1)

| ID | Approved policy | Implemented as | Rationale | Consequence |
|---|---|---|---|---|
| P-12 | The 2005–2006 Jolpica `race_time` values (e.g. "14:00:00Z") are **local clock times**, not UTC. They are read as local time at the circuit and converted to UTC with the IANA timezone database before weather alignment. The original value is never changed. | config `weather.race_time_local_clock_seasons: [2005, 2006]`; `race_start(..., local_clock=True)`; `start_time_source = "p12_local_clock"`; flag `p12_local_clock_applied`; info check `p12_local_clock_interpretation` | F-005: the same "14:00:00Z" across Melbourne, Tokyo, Europe and Montréal gives impossible UTC-based local starts (01:00, 23:00, 16:00, 09:00); local-start clustering R = 0.545/0.530 vs ≥ 0.833 in all other seasons | 37 races re-aligned; detector R after P-12 = 0.994 (2005) and 0.986 (2006), so no longer flagged; all 37 model-usable |

### F-006 · 2026-09-25 · Step 6.1 · P-12 applied and verified
- Re-run from cache: **0 network requests** (1,215 cache hits; the corrected UTC starts fall on the same request dates). 2,031 raw and interim files (Jolpica raw + manifest, Open-Meteo cache + logs, all non-weather interim CSVs) verified unchanged by SHA-256.
- **Only the 37 P-12 races changed.** The other 1,112 race rows are identical in every pre-existing column (one new column, `p12_local_clock_applied`), and their 14,313 hourly rows are identical. 481 hourly rows re-aligned.
- Corrected local starts: 14:00 for most races, Bahrain 14:30, Malaysia/Turkey 15:00, Canada 13:00, 2006 British GP 12:00 (World Cup schedule). Historical DST is handled correctly: Melbourne 6 Mar 2005 UTC+11 (DST) vs 2 Apr 2006 UTC+10 (DST already ended), Indiana 2005 UTC−5 (no DST observed) vs 2006 UTC−4.
- Quality report: **0 errors, 0 warnings.** Info only: 720 imputed pre-2005 starts, 37 P-12 interpretations, 3 local-date differences (Las Vegas 2023–2025, correct).
- **Usable weather: 1,138 of 1,138 eligible races** (1,149 − 11 Indianapolis 500). Start-time sources: 381 official UTC (2007–2025), 37 P-12 local clock (2005–2006), 731 imputed 14:00 local (1950–2004).

---

## Open assumptions to verify (from the blueprint)

| Assumption | Verify at | Status |
|---|---|---|
| Jolpica rate limit ≈ 4 req/s burst, 500 req/h | Step 2 | **Confirmed from official docs** (F-001); no rate-limit headers are sent |
| Open-Meteo free limits ≈ 600/min, 5,000/h, 10,000/day | Step 6 | **Confirmed** from the Open-Meteo terms (also 300,000/month; data CC-BY 4.0) |
| Official `race_time` is UTC from 2005 (Appendix A) | Step 6 | **Resolved by P-12 (APPROVED):** UTC from 2007; 2005–2006 values are local clock times, converted with the timezone database |
| `grid = 0` means pit-lane start | Step 7 | **Resolved by P-04:** pit-lane start only when the driver started and completed laps (48 rows); otherwise unknown or non-starter |
| `positionText` codes D, E, F, N meanings | Step 7 | **Resolved (F-003):** only R, W, D exist in 1950–2025. E, F, N never occur. W covers several non-start statuses |
| `position` orders retirees by laps completed | Step 7 | **Confirmed for 1950–2025 (F-004):** 8 inversions in 7,998 adjacent pairs, none after 2009 |
| circuitId `indianapolis` reused for 2000–2007 US GP | Step 7 | **Confirmed (F-003):** US GP 2000–2007 uses it. Filter the Indy 500 by race name |
| 1950s shared drives appear as separate rows | Step 7 | **Confirmed (F-003):** separate rows; driver_id repeats within a race (172 rows, 1950–1964) |
| Qualifying available from 1994 (blueprint §11.1, §13.2) | Step 7 | **Only partly (F-003):** 1994–2002 has qualifying for only some races; complete from 2003 |
| Constructor renames produce different constructorIds | Step 11 | **Confirmed for 2023→2024** (alfa→sauber, alphatauri→rb); earlier renames open. No ID ever has two names in the API |
| Open-Meteo Historical Forecast API exists (Future Work only) | — | Open |

## Earlier pending questions: **resolved by the approved policies**

| Question (raised in Steps 2–5) | Resolved by |
|---|---|
| Does a *classified* retirement count as DNF? | P-06: no; flagged `retired_while_classified` |
| DSQ detection must also use `status` (positionText can be numeric) | P-05: `result_category_detail` uses positionText **and** status |
| The non-starter filter must not use grid (DNS rows carry a grid value) | P-04 / P-08: `started` comes from the category, not from grid |
| Drivers absent from a race (no row) or listed without records | P-08: they never become model rows; no rows are invented |
| Duplicate (season, round, driver_id) rows and shared positions before 1965 | P-01 / P-02 |
| 1994–2002 qualifying gaps | P-03 |
