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

---

## Open assumptions to verify (from the blueprint)

| Assumption | Verify at | Status |
|---|---|---|
| Jolpica rate limit ≈ 4 req/s burst, 500 req/h | Step 2 | Open |
| Open-Meteo free limits ≈ 600/min, 5,000/h, 10,000/day | Step 6 | Open |
| `grid = 0` means pit-lane start | Step 7 | Open |
| `positionText` codes D, E, F, N meanings | Step 7 | Open |
| `position` orders retirees by laps completed | Step 7 | Open |
| circuitId `indianapolis` reused for 2000–2007 US GP | Step 7 | Open |
| 1950s shared drives appear as separate rows | Step 7 | Open |
| Constructor renames produce different constructorIds | Step 11 | Open |
| Open-Meteo Historical Forecast API exists (Future Work only) | — | Open |
