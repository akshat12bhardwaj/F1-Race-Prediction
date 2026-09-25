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

---

## Open assumptions to verify (from the blueprint)

| Assumption | Verify at | Status |
|---|---|---|
| Jolpica rate limit ≈ 4 req/s burst, 500 req/h | Step 2 | **Confirmed from official docs** (F-001); no rate-limit headers are sent |
| Open-Meteo free limits ≈ 600/min, 5,000/h, 10,000/day | Step 6 | Open |
| `grid = 0` means pit-lane start | Step 7 | **Partly:** one 2024 case is consistent with it; check full history at Step 7 |
| `positionText` codes D, E, F, N meanings | Step 7 | **Partly:** R, D, W seen in 2024 (W = "Did not start"); E, F, N not yet seen |
| `position` orders retirees by laps completed | Step 7 | **Confirmed for 2024** (24/24 races); check older eras at Step 7 |
| circuitId `indianapolis` reused for 2000–2007 US GP | Step 7 | Open |
| 1950s shared drives appear as separate rows | Step 7 | Open |
| Constructor renames produce different constructorIds | Step 11 | **Confirmed for 2023→2024** (alfa→sauber, alphatauri→rb); earlier renames open |
| Open-Meteo Historical Forecast API exists (Future Work only) | — | Open |

## Decisions pending (raised by Step 2, to be decided at Step 7, not yet changed)

| Question | Evidence |
|---|---|
| Does a *classified* retirement (status "Retired", numeric positionText) count as DNF? The blueprint §14 rule (positionText R/N) says no. | 6 such records in 2024 |
| DSQ detection must also use `status`, because positionText can be numeric for a disqualification | Belgian GP 2024, RUS |
| The non-starter filter must use status or positionText, not grid, because DNS records carry a grid value | GAS British GP, ALB/STR São Paulo GP 2024 |
