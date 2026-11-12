# Repository cleanup and publication exclusions

The current local implementation and existing Git history were preserved. Source, tests, model logic, canonical datasets, production model bundles, required historical comparison inputs, circuit geometry and final licensed assets remain available.

## Reviewed candidates

Before any deletion attempt, `CLEANUP_MANIFEST.csv` recorded each candidate path (or whole regenerable cache), file count, bytes and reason. Review found **548 files / 73.34 MiB** in the following categories:

| Category | Files | MiB | Reason |
|---|---:|---:|---|
| Superseded browser captures | 131 | 40.46 | Replaced in the repository by nine optimized real showcase images |
| Old development logs | 53 | 0.27 | Generated run output; scientific tables and protocols retained |
| Asset-research scratch | 179 | 9.15 | Intermediate candidates/contact sheets and one-off scripts; final assets and attribution retained |
| Caches | 148 | 10.11 | Recreated by Python, pytest and Vite |
| Duplicate pilot/subset datasets | 13 | 0.38 | Every row and column independently verified present in the canonical full snapshot |
| Duplicate model backups | 24 | 12.98 | SHA-256 identical to a retained bundle; removed publication paths unused by application/tests |

**Local deletion was blocked by automated safety review**, including a narrower file-by-file attempt. Consequently, no physical files were deleted and actual disk space recovered is **0 bytes**. The safe completed alternative is publication cleanup: old tracked logs were removed from the Git index while retaining their local copies; reviewed candidates are excluded by `.gitignore`. The manifest's `action` field explicitly records this distinction. No `git clean`, reset, force-push or remote-history rewrite was used.

## Deliberately retained

- `.venv-overhaul` is used by a running APEX instance; it was not treated as abandoned. It and `node_modules` are ignored, not published.
- Unique legacy rollback snapshots under `backups/` and `models/algorithms_v1_backup/` stay local and ignored; they are not needed by current runtime/tests but were retained rather than destroyed.
- `models/algorithms_d023_backup/` and `models/algorithms_wx/` are bundled because weather tests and the pre-registered selection workflow read them.
- `data/interim/subset_weather_race.csv` differs from the full weather table; it was retained rather than mislabeled a duplicate.
- Raw acquisition caches remain local. Required canonical interim/processed datasets are committed so a clone runs without collection.
- Asset research source records remain local. Published attribution comes from the final per-file manifests, credits pages and source notices.
- Scientific tables, EDA/model figures, notebooks, decisions and historical implementation notes were retained as coursework/reproducibility evidence. Old narrative test counts describe their dated milestones; current results live in `VERIFICATION.md`.
- `frontend/dist` remains available to the running app but is excluded; a fresh checkout builds it with `npm ci` and `npm run build`.

No required `.joblib` is silently omitted. All published files fit ordinary Git; no LFS or private artifact host is needed. Data/assets retain their individual licences; see `CREDITS.md`. The optimized nine-screenshot gallery totals about 1.15 MiB.
