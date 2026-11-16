# Reproducing APEX

## Run the bundled release

Use Python 3.11 and install `requirements-lock.txt`, the full resolved environment tested for publication. Direct dependencies remain documented in `requirements.txt` (especially scikit-learn 1.7.1, NumPy 2.2.6 and joblib 1.5.1). Use `npm ci` with `frontend/package-lock.json`, then `npm run build`. Launch `python -m uvicorn src.api:app --host 127.0.0.1 --port 8012` from the repository root. The README has complete PowerShell commands.

**No first-run training, API key or asset download is needed.** The published tree includes:

- Full canonical historical interim CSVs through 2025 and processed feature/snapshot tables.
- The production classification/regression bundles, density model, scenario models, legacy API models and chronological walk-forward bundles required by existing tests.
- D-033 weather candidate bundles and the D-023 comparison baseline required by weather-selection tests and the selection workflow. They are scientific comparison inputs, not unused backups.
- Local circuit geometry/elevation, five GLBs, driver portraits, constructor logos and their attribution/manifests.
- Scientific comparison, calibration and audit tables.

`python scripts/verify_bundle.py` checks every bundled data/model/public asset against SHA-256 and byte length in `docs/ARTIFACT_MANIFEST.json`. This uses only the Python standard library. Maintainers can regenerate the manifest with `--write` after intentionally updating and staging artifacts. `.gitattributes` preserves their exact bytes on checkout.

## Rebuild features and models intentionally

These commands write derived files. They are not part of normal startup, and publication does not retrain or alter the supplied models.

```powershell
python -m src.dataset
python -m src.features
python -m src.train
python -m src.snapshots
python -m src.driver_ratings
python -m src.train_scenario
python -m src.train_simulator
python -m src.train_algorithms
python -m src.unsupervised
```

Read each module's command-line help/docstring and D-019 through D-033 before repeating training. The current production course models use a per-model mixture of feature schemas chosen by D-033; simply running the default trainer does not reproduce that mixture. To reproduce its candidate-selection workflow:

```powershell
python -m src.train_algorithms --features v1 --out models/algorithms_wx/v1
python -m src.train_algorithms --features v2 --out models/algorithms_wx/v2
python -m src.select_weather_models
```

The selection rule uses chronological calibration/validation metrics, not the 2023–2025 test results. Preserve the supplied metadata and scientific tables if comparing reruns. Seed 42, feature definitions and calibration protocols are recorded in the model metadata and decisions log. Cross-era hypothetical races use retrospective form/peak selection and have no observed ground-truth accuracy.

## Reacquire historical data

Raw response caches remain local and are excluded from Git. Canonical interim tables are bundled, so neither runtime nor offline tests need those caches. For a deliberate new collection:

```powershell
python -m src.collect --start 1950 --end 2025
python -m src.policies
python -m src.weather
python -m src.validate_raw
```

The clients cache responses, retry temporary failures and observe configured request intervals. Initial historical collection can take hours. Upstream records may be corrected over time; a newly downloaded dataset is not guaranteed byte-identical to the bundled frozen snapshot. Review the [data terms and transformation attribution](../CREDITS.md) before redistribution or commercial use.

## Circuits and assets

The already-built `frontend/public/circuits.json` is the runtime source. `python -m src.circuit_geometry` can rebuild it from the included GeoJSON and Copernicus data; absent local DEM samples require network downloads. Layouts depict current/recent tracks, and only quality-gated profiles carry measured elevation.

Portrait/logo fetchers remain in `src/` with the reviewed selections. Third-party image availability can change; the published files and manifests are authoritative for this release. Keep per-file attribution and licence notices when redistributing derivatives.

## Browser verification

Windows checks use installed Edge through Playwright Core, without downloading another browser. Set `E2E_URL` to the running app. `npm run e2e:visual` additionally uses a Vite server on port 5177 for isolated material views; start it in another terminal with `API_PORT=8012` and `npm run dev -- --port 5177 --strictPort`. The GPU check requires usable hardware WebGL and opens a visible browser.

Screenshots can be regenerated using `npm run showcase`, then `python scripts/optimize_screenshots.py`. The capture script uses actual UI interactions and a real seeded 22-driver prediction. It never substitutes API responses.


## Replay the website model pairs on observed grids

Run `python -m src.evaluate_live_pair` from the repository root. The script uses the installed default and showcase pair configurations, calibrates on expanding 2017–2019 predictions, and refits only through 2022. It records model hashes and settings in `outputs/tables/live_pair_replay.json`. The 2023–2025 results are a chronological replay of previously examined races. The script also recomputes the legacy replay metrics with coherent podium probabilities in `historical_coherent_replay.csv`. Keep the first-run frozen table as an archival record. Development can continue; validate changed configurations on future races for an independent accuracy claim.
