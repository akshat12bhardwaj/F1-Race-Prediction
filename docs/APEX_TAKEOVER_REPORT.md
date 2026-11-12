# APEX takeover completion — 12 November 2026

## Recovered state

The local working directory was retained as the source of truth. Most application code, model artifacts, processed data and tests were already untracked; the tracked README and decisions log also contained earlier uncommitted work. No reset, clean, commit, push, dataset regeneration or model retraining was performed. No AGENTS.md or CLAUDE.md was found in the repository or the checked parent instruction locations.

The latest edits establish where development stopped: `simulate.py` already contained the D-036 peak selection and 22/11/2 validators; `api.py` exposed peak teams and the new default grid; the editor already had A–Z search-only lists and 22 slots; `test_peak_grid.py` existed; the most recent file was `frontend/e2e/app.mjs`, partially converted to the new workflow. D-036 had not been recorded in the decisions log. Existing progress documents still described older milestones.

Port 8012 already ran the new backend. Ports 8010 and 8011 served older instances. All three existing servers were left running. The frontend was rebuilt, and the live 8012 peak/default-grid responses were compared against local source results. Its served index was compared byte-for-byte with `frontend/dist/index.html`.

## Completion table

| Requirement | Initial state | Final state | Verification |
|---|---|---|---|
| 22-driver grid | Implemented locally, not fully verified | Exactly 22 unique drivers; P1–P22 | Python peak-grid/API tests; 118-check browser journey |
| 11 constructors per race | Backend/editor rules present | Invalid team counts block prediction | Python rejection tests; browser creates and repairs a 12-team grid |
| 2 drivers per constructor | Seat assignment present | Capacity indicators, full-team seat swaps, final validation | Pure-function and browser assignment tests |
| Full historical constructor pool | Existing records intact | 65 constructors, 537 eligible car-seasons, 208 eligible drivers | Independently recomputed eligibility/peak tests; live API |
| Peak constructor performance | Selection and real-snapshot integration present | One reproducible coherent eligible season per constructor; documented | Independent peak recomputation; snapshot-column comparisons |
| Alphabetical driver list | Implemented | Full displayed names A–Z | Unit and browser ordering assertions |
| Alphabetical constructor list | Implemented | 65 unique constructor names A–Z | Unit and browser ordering assertions |
| Driver/constructor search | Implemented | Name searches ignore case and accents | Unit tests; browser checks including Räikkönen/Pérez |
| Removed era/year dropdowns | UI removal done; old browser tests depended on them | No editor selectors; historical records retained | Browser DOM assertions; repaired portrait/logo suites |
| Drag-and-drop assignments | Implemented, verification incomplete | Driver replacement/move/removal and constructor seat swaps work; Undo/Redo retained | Full browser journey and lineup unit tests |
| ML prediction compatibility | Variable-field models already installed | All 10 classifiers and 6 regressors work with 22 peak-car entries in dry/wet conditions | Artifact-level API compatibility tests; all four requested hypothetical pairings retain separate driver/car features |
| 22-car 3D race | Rendering already driven by entry count; scripts expected 20 | 22 loaded GLB cars, staggered start, synchronized timing and full final ranking | GPU, asset, motion/collision and camera browser suites |

## Work completed during takeover

- Added frontend final-grid validation for duplicate driver IDs and cars outside the eligible peak list, with regression assertions.
- Finished the interrupted app browser test: wait for lazy logo decoding, compare predictions with the actual edited grid/peak season, and give onboard camera testing its own playback interval so the race cannot finish before Pause is tested.
- Updated GPU and camera checks to 22 cars. Expanded randomized ordering and collision tests from 20 to 22 cars.
- Updated logo and portrait browser tests to search-only constructor/driver lists. Logo assignments use actual peak seasons; tests restore a valid grid before prediction. Existing manifests and era-specific marks are preserved.
- Reworked asset browser verification around a valid 22-driver/11-constructor fixture covering all five existing body assets. It tests cached downloads, Follow/Onboard, final standings and failed-asset handling for all 22 identities.
- Added dry/wet compatibility checks exercising every selectable ML artifact, plus direct feature comparisons for Senna–Mercedes, Schumacher–McLaren, Hamilton–Ferrari and Verstappen–Williams.
- Installed the two already-pinned missing notebook dependencies (`ipykernel==6.29.5`, `jupyter==1.1.1`) in `.venv-overhaul`; this repaired the two environment-test failures without changing requirements or model packages.
- Rebuilt the frontend, updated the README's current workflow and runnable prediction example, and recorded D-036 and this handoff.

## Peak-car methodology

For each eligible constructor-season, use the historical season's average qualifying-grid percentile and finishing percentile (0 is best, 1 is last), and its DNF share minus the contemporaneous field's DNF share:

`peak_score = -(0.40 × mean_grid_percentile + 0.40 × mean_finish_percentile + 0.20 × (team_DNF_share − season_field_DNF_share))`

The grid percentile is a qualifying-performance proxy; it is not a raw qualifying lap-time comparison. Finishing percentile summarizes race performance, and the DNF term adjusts reliability against the constructor's own era. Win totals are not selection criteria. Prefer eligible seasons with at least eight races; if a constructor has no such season, retain it using its eligible seasons and mark `low_evidence`. Pick the highest unrounded score, breaking exact ties toward the newer season. The response records the chosen season, component metrics, races, score, candidate count, evidence flag and runner-up.

Examples from the actual current data: McLaren 1988, Ferrari 2002, Mercedes 2015, Williams 1993, Red Bull 2011, Brawn 2009. These are the output of the stated rule, not manually chosen famous cars.

Only the selected season's existing real constructor snapshot is fed into the ML model. No per-feature cross-season composite is made. Driver features continue to come from the driver's existing validated form season. The default lineup uses 11 high-scoring constructors with two available, unique eligible drivers who actually drove that constructor's selected car-season; each driver still uses their independent form season.

This selection is retrospective through 2025 for hypothetical races. It is not inserted into chronological historical training or evaluation. Existing model weights, feature schemas, calibration, random seeds and train/validation/test splits are unchanged. No predictive-accuracy improvement is claimed; cross-era race outcomes have no factual ground truth.

## Verification

Detailed logs are under `outputs/logs/takeover-*.log`.

- Initial offline Python run: 263 passed, 2 failed, 1 live test deselected. Both failures were missing notebook dependencies; after installing the pinned packages, all 7 environment tests passed.
- Peak-grid/model compatibility suite: 18 passed, including 20 full-field dry/wet API predictions covering all 16 selectable artifacts. Additional requested-pairings feature test: 1 passed.
- Frontend unit tests: 43 passed, 0 failed. Includes 22-car randomized overtake ordering and dense non-overlap sampling.
- Production frontend build: passed.
- Main headless Edge journey: 118 checks passed; no browser/API/asset errors; measured 10.94 seconds of active playback in the instrumented run (includes frame/automation overhead).
- Logo browser suite: 79 checks passed; all 65 peak constructor cards checked. 37 peaks have a verified era-compatible logo; the rest correctly use badges. Other historical logo variants remain stored.
- Portrait browser suite: all 5 most recently added portraits rendered correctly; all 14 missing-photo monograms and failed-image fallback passed; actual coverage 194/208.
- Asset browser suite: passed; five GLB body assets across a valid full 22-car field, cached loading, camera tracking, prediction/final-render agreement and failure fallback.
- Real GPU: NVIDIA GeForce RTX 4060 Laptop GPU, ANGLE/Direct3D11. All checks passed; approximately 240 FPS in Spa Overview, Spa Follow, Monaco Onboard and heavy rain; worst sampled frames 4–5 ms.
- Camera/visual browser suite: 231 checks passed across Spa, Suzuka, Monaco and Monza; zoom/pan limits, terrain clearance, mobile resize, follow/onboard, pause/resume, timing synchronization, touch pinch and 20 isolated era/angle material screenshots.
- Manually inspected generated editor, results and GLB follow screenshots. The three-column layout, all 22 visible editor slots, dark/lime theme, badges/portraits and 3D views are preserved. Main browser run also checked 390 px mobile width with zero horizontal overflow.

Final offline Python rerun: **267 passed, 1 live test deselected, 0 failures**, in 137.29 seconds (`outputs/logs/takeover-python-final.log`). The four-requested-pairings feature test was added after that run collected its tests and passed separately (`outputs/logs/takeover-pairings.log`): **268 distinct Python tests verified in total**.

## Files edited during takeover

- `frontend/src/lineup.js`, `frontend/src/App.jsx`
- `frontend/e2e/app.mjs`, `gpu.mjs`, `solid-camera.mjs`, `car-assets-browser.mjs`, `logos.mjs`, `portraits.mjs`, `lineup.test.mjs`, `motion.test.mjs`
- `tests/test_peak_grid.py`
- `README.md`, `docs/DECISIONS_LOG.md`, `docs/PROGRESS.md`, `docs/APEX_TAKEOVER_REPORT.md`
- Generated production bundle under `frontend/dist/`, verification logs, screenshots and existing browser asset-review outputs.

The peak-selection/backend implementation itself was already present and was preserved. No historical data, constructor identities, portraits, logos or model artifacts were removed or rewritten.

## Running application and PowerShell commands

Working URL: **http://127.0.0.1:8012** (already running). Do not launch another process on this occupied port. To launch after the existing instance has been stopped intentionally, use:

```powershell
Set-Location F1-Race-Prediction
Push-Location frontend
npm run build
Pop-Location
& .\.venv-overhaul\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8012
```

To reproduce checks against the running app:

```powershell
& .\.venv-overhaul\Scripts\python.exe -m pytest -m 'not live' -q
Push-Location frontend
$env:E2E_URL = 'http://127.0.0.1:8012'
npm run e2e
npm run e2e:logos
npm run e2e:portraits
npm run e2e:assets
npm run e2e:gpu
npm run e2e:visual
Pop-Location
```

The isolated material views used by `e2e:visual` additionally use the already-running Vite instance on port 5177. If that instance is absent, start `npm run dev -- --port 5177 --strictPort` from `frontend` in a separate terminal, setting `$env:API_PORT='8012'` first. No unrelated process was terminated.

## Remaining limitations

No unresolved functional blocker was found in the requested workflow. Existing historical GLBs remain simplified era approximations, and 2010–2017 uses the disclosed 2000s body; the detailed ground-effect asset is used for 2022–2025 cars. Historical identities without a verified peak-era logo or free portrait retain explicit badge/initial fallbacks. Cross-era comparisons are experimental and overtakes are illustrative. Existing dependency deprecation and sparse-history/imputation warnings remain; they did not fail the application checks. The live internet collection test is intentionally excluded from offline verification.
