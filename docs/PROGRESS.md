# Progress summary (Phases 1–10)

**Status (2026-11-06): complete except the test evaluation, which was done on 2026-11-07 (see below).** Machine-readable state: `AUTONOMOUS_PROGRESS.json`.

| Phase | Deliverable | Verification |
|---|---|---|
| 1 Dataset | `src/dataset.py` → `data/processed/master_base.csv` (25,873 rows; 25,066 model rows; 1,138 races) | `tests/test_dataset.py` (integrity: unique keys, one winner per race, 1..k targets, pit-lane grid) |
| 2 Features | `src/features.py` → 46 T0-safe features | `tests/test_leakage.py` T1 recompute-from-past (exact), T2 perturbation, T3 cold start, T4 AUC suspicion, T5 split order, T6 denylist |
| 3 Evaluation | `src/splits.py`, `src/evaluate.py` (row, race, bootstrap metrics, coherence) | `tests/test_evaluate.py` (incl. row-order leak regression test) |
| 4 Models | 8 classifiers × 2 tasks, 9 regressors (incl. baselines + NumPy GD logistic) | `outputs/tables/model_comparison.csv`, paired bootstrap |
| 5 Engine | `src/predict.py` + CLI; walk-forward models per season | API tests: Σwin = 1, Σpodium = 3, unique order, identical repeats, outcome-scrambling invariance |
| 6 Analysis | `src/analysis.py`, 7 figures, `docs/LITERATURE_REVIEW.md` (10 verified sources) | tables in `outputs/tables/` |
| 7 Backend | `src/api.py` (FastAPI) | `tests/test_api.py` (13) |
| 8 Frontend | `frontend/` React + Vite + Plotly | `npm run build` |
| 9 Integration | site served by FastAPI | `npm run e2e`: 26/26 checks in headless Edge, screenshots in `outputs/figures/ui/` |
| 10 Validation | full suite | **142 passed** (`pytest -m "not live"`) |

## Key results (walk-forward 2017–2022, 123 races, out-of-sample; test seasons untouched)

- Final system: winner top-1 **50.4%** (95% CI 41–59%), winner in top 3 **89.4%**, podium overlap **70.2%** (66–75%), race Spearman **0.64**, ranked position MAE **3.45** places.
- Winner (random forest) has significantly better winner log-loss than the grid heuristic (Δ −0.20, CI −0.34 to −0.07). It is not distinguishable from logistic regression or gradient boosting.
- Position (ridge): significantly lower race MAE than grid order (Δ −0.18 places, CI −0.31 to −0.06). Linear models tie with each other and match or beat the tree ensembles.
- Podium: PR-AUC improves (0.74 vs 0.67 for the grid heuristic), but race-level **podium overlap is not distinguishable from grid order**.
- Ablation (2022): grid/qualifying is by far the most important group. Race-window "oracle" weather adds nothing over pre-start weather.

## E10 test evaluation

Done on 2026-11-07 after the owner's authorisation (see the update below and D-020).

---

# Update 2026-11-07: custom race simulator (COMPLETE)

The project owner removed the freeze restrictions and asked for a custom race simulator. Status: **complete and verified**.

| Item | Verification |
|---|---|
| Scenario features, driver/constructor snapshots, circuit catalogue | leakage tests on the new columns; snapshot equals training features (`tests/test_simulate.py`) |
| Plackett–Luce ranking model (truncated top-1, time-weighted) | gradient checks, weight recovery, IIA/subset property (`tests/test_ranking.py`) |
| `predict_custom_race` + 6 new endpoints | 23 simulator tests: validation, coherence, seed reproducibility, sensitivity to grid, circuit, weather, snapshot and car |
| Website: Simulator (default), Historical replay, Model comparison | `npm run e2e`: 57/57 browser checks |
| 2023–2025 | evaluated (v1 clean hold-out; scenario model post-hoc, D-020); replay unlocked |

All offline tests: **171 passed**.

## Latest handoff — 12 November 2026

The interrupted 22-driver / 11-constructor / automatic-peak-car work has been completed and verified. Existing local implementation and historical records were preserved. Current app: **http://127.0.0.1:8012**. The historical milestone notes elsewhere in this file describe earlier versions; the current completion table, repaired tests, counts, GPU results, model compatibility and launch instructions are in [APEX_TAKEOVER_REPORT.md](APEX_TAKEOVER_REPORT.md). No model retraining, commit or push was performed.


# Update 2026-11-07 (2): full-grid simulator with prime form and customisable cars (COMPLETE)

| Item | Verification |
|---|---|
| Default full grid at circuit capacity (≤ 26), prime-form drivers, prime cars | `tests/test_simulate.py` (capacity, prime rule, defaults); browser: opens with 26 cars at Monaco |
| Car-neutral driver skill (teammate-network ratings, leakage-safe) | `tests/test_driver_ratings.py` |
| Car swap swaps exactly the car inputs; predictions move the right way; reproducible | `test_swapping_cars_swaps_exactly_the_car_inputs`, `test_car_swap_changes_predictions_coherently_and_reproducibly`; browser swap Senna ↔ Hunt |
| Grid estimated by the qualifying model (no manual grid) | estimated grid is a permutation; UI has no grid controls |
| All tests | 172 pytest, 56 browser checks |


# Update 2026-11-07 (3): interactive ML algorithm comparison (COMPLETE)

18 algorithms (all supervised algorithms of the course handout + Naive Bayes, MLP, Plackett–Luce, grid baseline) are trained under one protocol and run on the same custom race with shared random numbers. Clustering and density estimation have non-predictive roles. Verified by `tests/test_algorithms.py` (21) and `frontend/e2e/algorithms.mjs` (25 checks). All tests: 193 pytest, 81 browser checks.
