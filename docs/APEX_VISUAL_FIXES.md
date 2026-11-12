# APEX geometry and camera corrections

Implemented and visually checked on 11 November 2026. No ML/backend source, trained model, historical mapping, constructor palette or race-motion function was changed. Nothing was committed or pushed.

## Hollow cars: cause and repair

The source Formula F222 GLB contains inward-facing bodywork, wing and wheel surfaces. Backface culling therefore removes exterior panels; its main body shell is 7,056 triangles. The batching importer also baked reflected transforms without preserving triangle winding. Opaque source materials were not the cause. An overly distant Overview near plane could additionally slice cars during close manual inspection.

`carGeometry.js` repairs the audited source meshes' triangle winding and/or normals before batching, and preserves face orientation when baking a negative-determinant transform. Repairs are applied to the decoded, cached original geometry. Original GLB files remain unchanged; no assets were replaced and no procedural geometry was substituted. The 59,884 original triangles are preserved. Normal, signed-volume and exterior-ray tests verify the actual asset, including rays hitting the near sidepod surface rather than the far interior. Source-specific corrections are pinned by the original binary checksum in tests. The open cockpit, halo and genuine intake boundaries remain open. Materials remain opaque and front-sided; DoubleSide is not used to conceal defects. The existing shared geometry, five meshes/car, constructor paint and wheel animation remain active.

The camera now uses a close-safe adaptive near plane (0.04-0.25 m). The Onboard eye/look direction also shows the nose and front wheel/suspension edges rather than pointing above all the bodywork.

## Zoom-out and panning: cause and repair

The old maximum distance was a multiplier of two fixed fitted viewpoints. Orbiting changes the projected circuit size, panning a target anywhere inside the full world bounding box can put the track off screen, and scripted cameras bypass OrbitControls distance clamping. During resize, the old controls limit could also undo new framing. Wheel/drag handoff from tracking happened after OrbitControls had already processed the input.

`cameraBounds.js` solves perspective frustum inequalities using the road/runoff bounds, sample depths, current view direction, FOV and aspect ratio. It recentres the projected bounds and calculates a direction-dependent maximum. Wide-view targets use a small screen-space pan budget; close manual views stay anchored to the selected car/nearby road. Targets remain within the measured terrain footprint. Close manual Follow/Onboard views have a 12-car-length maximum range. Hard constraints run after orbit/zoom, scripted transitions and tracking, before rendering. Reset, circuit changes and resize recompute framing; resize updates the controls limit before it can reapply the old value. Capture-phase input handoff prevents a first wheel/touch/drag from using a stale tracking target. Terrain clearance remains enforced. No terrain/background enlargement or fog workaround was added.

HD-only 3D rendering, 1:1 elevation, Overview/Follow/Onboard/Top/Reset, the ten-second replay, start lights, timing, pause/resume and constructor/season selection remain intact.

## Changed files

- `frontend/src/car.js`: uses the audited geometry repair during cached loading.
- `frontend/src/carGeometry.js`: source orientation repairs and reflection-safe baking.
- `frontend/src/cameraBounds.js`: perspective fit and continuous camera constraints.
- `frontend/src/components/Track3D.jsx`: framing, input handoff, adaptive clipping and constraint integration.
- `frontend/e2e/car-geometry.test.mjs`: actual-GLB exterior, volume, normal, reflection and checksum regressions.
- `frontend/e2e/camera-bounds.test.mjs`: four-circuit/direction/aspect, pan, local-view and near-plane regressions.
- `frontend/e2e/solid-camera.mjs`: hardware browser gestures, resize, touch, four full-grid pause/resume races and visual capture.
- `frontend/e2e/car-inspection.html` and `car-inspection.js`: isolated multi-era/front/rear/side diagnostic fixture.
- `frontend/package.json`: runs the new regressions and exposes `e2e:visual`.
- `frontend/public/models/ATTRIBUTION.html`: records runtime geometry repairs.
- This note and a link from the earlier overhaul report.

## Actual results

- Production build passed.
- 36 frontend unit tests passed, including 10 new geometry/camera tests.
- All 107 full-journey browser checks passed, including historical/cross-era editing, ten-second replay, start lights, weather, timing synchronization and mobile.
- All 231 additional hardware visual/camera checks passed on Spa, Suzuka, Monaco and Monza: extreme wheel zoom/pan in all four modes, resize/circuit switching, two-finger pinch, terrain clearance, four 20-car races, paused timing synchronization and resumed predicted finishing order.
- RTX 4060 Laptop GPU: 240 FPS in Spa Overview/Follow and Monaco Onboard; heavy rain 239 FPS. Overview: 108 draw calls, 1,244,172 triangles. Last sampled slowest frames: 4-5 ms in dry views, 12 ms in rain. These are browser animation-frame measurements during active rendering.
- Inspected rendered Overview/Top and maximum-pan images across all four circuits; inspected modern side/front/rear/three-quarter/top and historical body views. The repaired modern exterior is solid, and circuits remain prominent at the tested limits. Captured 20 isolated era-angle images plus main-app and before/after screenshots under `outputs/figures/ui/solid-camera/`.
- Logs: `outputs/logs/solid-camera-e2e.log`, `solid-camera-gpu.log`, `solid-camera-visual.log`, and `solid-camera-visual.json`.

Before/after comparison: `before-batched-side.png` versus `after-batched-side.png`; the source's hollow shell is also visible in `before-source-side.png`. Main-app checks include `after-spa-overview-max-pan.png` and corresponding Suzuka/Monaco/Monza files.

## Limits and launch

Historical model quality remains as previously disclosed: simplified era bases, with a 2000s carryover for 2010-2017. The fixes do not turn those assets into exact historical chassis. Backend tests were not rerun because no backend or shared prediction/replay logic changed; browser checks exercised the existing prediction API.

Working application: http://127.0.0.1:8011 . This server is already running; refresh the browser. The older process on 8010 was left untouched.

```powershell
Set-Location F1-Race-Prediction
npm --prefix frontend run build
& "$env:USERPROFILE\.venvs\f1-race-prediction\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8011
```

For the visual regression fixture, run Vite in another terminal before `npm run e2e:visual`:

```powershell
Set-Location frontend
$env:API_PORT = '8011'
npm run dev -- --host 127.0.0.1 --port 5177 --strictPort
```
