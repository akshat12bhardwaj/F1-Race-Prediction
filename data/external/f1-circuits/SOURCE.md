# F1 circuit geometry (visualisation only)

- Source: https://github.com/bacinger/f1-circuits (`f1-circuits.geojson`, downloaded 2026-11-11), MIT licence (see `LICENSE`).
- Content: one GeoJSON LineString per circuit, tracing the **current/recent** layout from map data, with the official length.
- Use: the 3D track view only (`python -m src.circuit_geometry` → `frontend/public/circuits.json`). It is **not** a model
  feature. Historic eras may have used different layouts, so the app labels these shapes "current layout".

# Circuit elevation (visualisation only)

- Source: Copernicus DEM GLO-30 (30 m), public tiles on AWS Open Data (https://registry.opendata.aws/copernicus-dem/),
  © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
  Free use with this attribution.
- Method (`python -m src.circuit_geometry`): bilinear samples along each centreline and on a 40×40 terrain grid (each node the
  mean of 5×5 samples), cached in `data/raw/dem/`. The DEM is a surface model (trees, buildings), so the track profile is
  smoothed (Gaussian, σ 60 m). Circuits whose residual noise or steepest grade is implausible are drawn flat and labelled.
- Azerbaijan (Baku) has no public 30 m tile, so Baku uses the 90 m release (GLO-90) of the same DEM.
- Limits: no banking data (level cross-section); bridges and tunnels follow the ground (e.g. the Suzuka crossover).
