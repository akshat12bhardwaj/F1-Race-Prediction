> Follow-up: source face orientation and camera limits were corrected after this original handoff. See [current geometry/camera fix results](APEX_VISUAL_FIXES.md).

# APEX 3D car and circuit overhaul

Implemented and checked locally on 11 November 2026. The original box-built procedural car renderer has been replaced by local GLB assets. The modern asset meets the detailed-model objective; historical visual quality remains a documented gap. No models were retrained, no prediction code was changed, and nothing was committed or pushed.

## A. Asset inventory

| Car season | Implemented local GLB | Source / author | License | Implemented / quality |
|---|---|---|---|---|
| 1981-1989 | `frontend/public/models/historical/15176.glb` | [Turbo flat-bottom](https://3dassets.dev/assets/grand-prix-circuit-and-open-wheel-turbo-flat-bottom-7e7ade37), 3D Assets | CC0 1.0 | Yes, simplified generic approximation; 386,940 bytes / 8,864 triangles |
| 1990-1999 | `frontend/public/models/historical/15173.glb` | [High-nose](https://3dassets.dev/assets/grand-prix-circuit-and-open-wheel-high-nose-racer-bbddbadd), 3D Assets | CC0 1.0 | Yes, simplified; very similar to 1980s base; 386,912 bytes / 8,864 triangles |
| 2000-2009 | `frontend/public/models/historical/15177.glb` | [Narrow-track](https://3dassets.dev/assets/grand-prix-circuit-and-open-wheel-narrow-track-racer-565b5129), 3D Assets | CC0 1.0 | Yes, simplified generic approximation; 375,312 bytes / 9,728 triangles |
| 2010-2017 | Reuses 15177.glb | No independent base acquired | CC0 1.0 carryover | Explicitly labelled 2000s carryover; distinct 2010s silhouette unavailable |
| 2018-2021 | `frontend/public/models/historical/18685.glb` | [Generic halo racer](https://3dassets.dev/assets/transport-collection-hd-open-wheel-racer-aa0e5cef), 3D Assets | CC0 1.0 | Yes, simplified halo-era approximation; 357,144 bytes / 8,028 triangles |
| 2022-2025 | `frontend/public/models/formula-2022.glb` | [Formula F222 Ultimate](https://sketchfab.com/3d-models/formula-f222-ultimate-de780743451748d6b5567b1e190c7908), CarlosCG31; [licensed optimized distribution](https://github.com/BadCodesGG/interactive-f1) | CC BY 4.0 | Yes, detailed 2022-era base; 446,632 bytes / 59,884 triangles |

Five unique GLBs total 1,952,940 bytes (1.86 MiB). All are self-contained. The 3D Assets supplier discloses AI assistance; these historical files are not premium historical chassis scans. Full source records, checksums, license texts, modifications, original creator attribution and unavailable candidates are retained in `frontend/public/models/`. The visible 3D Credits link opens `/models/ATTRIBUTION.html`; the existing Credits page also links to it.

## B. Visual changes and loading

- `frontend/src/car.js` now uses Three.js GLTFLoader with the bundled local Meshopt decoder. The modern GLB contains shaped bodywork, wings/endplates, suspension, cockpit/halo, rims and separate wheel assemblies. Historical bases contain simpler shaped bodies, wheels/rims and suspension; no false claim of exact historical geometry is made.
- Templates are cached by asset URL, including in-flight promises. The initial grid preloads its required assets. Geometry is shared by every car using a template and preserved across circuit/grid changes. Static assemblies are merged into one mesh, plus four independent wheel meshes: five draw calls per visible car, before scene objects. Each car owns its material, so team colours never bleed between entries.
- Original material colours, roughness and metallic response are retained for rubber, metal and carbon surfaces; paintable regions receive an original three-colour pattern: primary body, side panels and narrow stripes. No external textures are fetched. These are approximate liveries, without exact sponsor decals.
- Models normalize to +Z forward / +Y up, centre in X/Z, ground at Y=0, and fit within the existing 5.6 x 1.9 m visual envelope. Visual scale is a modest 1.15x, independent of model selection. Simulation spacing logic still uses its existing constants; the GLB selected never changes prediction or race trajectories.
- Separate wheel assemblies roll according to absolute replay distance (including scrub/reset) and front wheels steer from local track curvature. Cars retain the existing slope pitch. The road cross-section is flat because measured banking is unavailable; no banking data is invented.
- Overview searches circuit-specific camera headings and fits the full circuit to the current viewport, including resize. Zoom-out is capped to 1.12x the fitted maximum camera distance, panning is bounded to the circuit footprint, and free-camera terrain clearance is retained.
- Follow now sits approximately 2.3 car lengths behind the selected car. Onboard sits above the cockpit/halo, about 0.24 car lengths from the road anchor; own-car selection ring/tag are suppressed in that view.
- 2D routing/component, quality selection, Lite and automatic Lite fallback, HD toggle, and height exaggeration controls/state were removed. Rendering stays HD; measured elevation is 1:1. Existing explicitly labelled approximate crossover bridge clearance remains; it is not measured DEM detail.
- Loading progress and failures are visible. Failed assets retain driver positions/markers and camera controls; there is no procedural-car or alternative-quality fallback. Failed cache entries are evicted for a subsequent load attempt.

## C. Constructor and season mapping

`frontend/src/carAssets.js` owns visual selection. It reads `constructor_id` and `constructor_season`, never the driver's form season. Season boundaries are 1990, 2000, 2018 and 2022, with the explicit 2010-2017 carryover gap above. The same era base can serve several constructors, with separate original colour patterns. Examples include white/red historical McLaren versus silver/red or papaya modern McLaren; red/white Ferrari; black/turquoise or earlier silver Mercedes; navy/red/yellow Red Bull. Other constructors use their existing identity colour plus contrasting secondary and stripe colours. Exact team chassis and liveries are not claimed.

A 1988 Senna placed in a 2025 Red Bull renders the modern Red Bull-coloured GLB; a 2025 Norris in a 1991 McLaren renders the 1990s base. Driver identities, predicted order, probabilities and timing data are unchanged.

## D. Measured performance

GPU: ANGLE / Direct3D11 on NVIDIA GeForce RTX 4060 Laptop GPU, visible Edge browser, 1600 x 960 viewport, permanent HD. The dedicated GPU test samples browser animation frames during active 20-car playback; these are observed frame rates, not GPU-timer profiling or a guarantee for every device/circuit.

| Scenario | Observed FPS | Slowest sampled frame | Draw calls | Triangles |
|---|---:|---:|---:|---:|
| Spa Overview | 240 | 4 ms | 108 | 1,244,172 |
| Spa Follow | 240 | 5 ms | 12 | 106,446 |
| Monaco Onboard | 240 | 5 ms | 6 | 73,598 |
| Heavy rain | 240 | 5 ms | Not separately sampled | Not separately sampled |

Lower close-view counts result from frustum culling, not a Lite mode. GPU checks passed. See `outputs/logs/gpu_overhaul.log`.

A fresh five-era browser page loaded all five GLBs successfully; the recorded run reached ready in approximately 0.74 s including app/API initialization. Individual local GLB transfers took roughly 34-36 ms. These local measurements depend on machine and cache state. Switching camera modes triggered no additional GLB requests. The default contemporary grid requests only one unique GLB. Source geometry is decoded once per asset and reused; no CDN, paid download or sign-in is required at runtime.

## E. Verification

- Build: passed `npm run build`.
- Python: **245 passed, 1 deselected** (`not live`), 119 s, in the existing `an isolated Python 3.11` environment. Warnings concern existing matplotlib deprecations and missing sprint features; no failures. The system Python 3.13 environment is incompatible with the saved scikit-learn 1.7.1 artifacts and was not used for the passing run.
- Frontend: **26 unit tests passed**, including eight new asset/mapping/license/local-dependency tests.
- Full browser journey: **107 checks passed** on hardware rendering. It covers historical/cross-era editing and payloads, five-red-light start, grid alignment/spacing/road anchor, manual orbit/pan/zoom/terrain clearance, reset, zoom-out bounds, Follow, cockpit Onboard, scrubbing, pause/resume, weather, result/timing synchronization, mobile and absence of removed controls. Active replay measured **10.02 seconds**, excluding inference and automation pauses. No unexpected browser/API/asset errors. See `outputs/logs/e2e_overhaul.log`.
- Dedicated era integration: all five locally loaded model IDs and constructor palettes were checked for Senna/McLaren 1988, Schumacher/Ferrari 2004, Hamilton/Mercedes 2020, Verstappen/Red Bull 2023, and Norris/McLaren 1991. Five meshes/car, one request/unique file, camera switching cache reuse and final predicted-order synchronization passed. An intentionally blocked-GLB test also passed: all 20 driver identities and the camera controls remain usable, with a visible failure message. Source evidence and screenshots: `outputs/logs/car-assets-browser.json`, `outputs/logs/car-assets-browser.log`, and `outputs/figures/ui/overhaul-*.png`.
- Visual inspection: inspected Overview, modern Follow/Onboard and all required era examples. Modern geometry is substantially more detailed, orientation is correct, starting cars sit on the road with visible clear space between them, and circuit framing fills most of the viewer. Historical simplified construction remains clearly visible. The tests verify road anchors and spacing; they are not an exhaustive proof against every visual contact defect on all 77 circuits.

## F. Remaining limitations

1. The detailed-quality requirement is satisfied by the contemporary base, **not across the full 1981-2025 range**. Premium-quality 1980s/1990s/2000s/2010s assets remain unavailable in the acquired set. The historical CC0 files are simplified, the 1980s/1990s silhouettes are too similar, and 2010-2017 lacks its own base. These gaps are disclosed in-app when selected and in the credits.
2. No season-specific constructor chassis or exact historic sponsor liveries are claimed. The 2022 base approximates 2022-2025 cars; it does not reproduce each team's design evolution. Better historical candidates (MP4/4, F2004, Ferrari 150 Italia/SF90) require authenticated downloads, and no restrictions were bypassed.
3. Cars remain physically small at whole-circuit scale. Screen-size constructor markers aid recognition; Follow/Onboard reveal the detail without enlarging vehicles to track-sized proportions.
4. Measured bank angles are unavailable. Existing DEM and approximate crossover bridge information retain their limitations. Wheel rotation and steering illustrate replay motion rather than simulate vehicle dynamics.
5. Python's default interpreter on this computer differs from the project environment. Use the exact environment command below for tests and launch.
6. A temporary `.venv-overhaul/` verification environment was created before the existing environment was located. Automatic approval review blocked its recursive cleanup with the reason `blocked by policy`; the directory is retained and ignored by git. It is not used by the app. No trained artifacts were changed.

## G. Launch and test commands (PowerShell)

Current site: **http://127.0.0.1:8010** (responding successfully). The verification server is also running at http://127.0.0.1:8011. Refresh an already-open browser tab to load the new build.

```powershell
Set-Location F1-Race-Prediction
npm --prefix frontend run build
& "$env:USERPROFILE\.venvs\f1-race-prediction\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8010
```

Port 8010 is already serving this project; a second server on that port will report it is in use. Start the command only after stopping the existing server, or use another port. No new server is needed to view the current build.

```powershell
# From the project root, with the existing verification server on 8011:
& "$env:USERPROFILE\.venvs\f1-race-prediction\Scripts\python.exe" -m pytest tests/ -q -m 'not live'
Set-Location frontend
$env:E2E_HARDWARE = '1'
npm run e2e
npm run e2e:assets
npm run e2e:gpu
```
