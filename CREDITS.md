# Data and asset credits

APEX is an independent educational project, unaffiliated with Formula 1, the FIA, constructors or drivers. Names and marks identify their respective owners.

## Historical race data

- Results, grids, qualifying, sprint records, drivers and constructors: [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the Ergast-compatible community API, historical collection through 2025. The [API terms](https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md) specify **CC BY-NC-SA 4.0**. See [the licence](https://creativecommons.org/licenses/by-nc-sa/4.0/). Included race-data snapshots and their derived feature tables retain those non-commercial, attribution and share-alike conditions; publication here does not grant commercial data rights. Transformations include policy flags, era-relative features, driver/constructor snapshots and teammate-network ratings; definitions are in `src/` and `docs/DECISIONS_LOG.md`.
- Weather data by [Open-Meteo.com](https://open-meteo.com/), [CC BY 4.0](https://open-meteo.com/en/license). Historical reanalysis is sampled around race starts; derived pre-start summaries and model inputs are described in the decisions log. Reanalysis is not an archived operational forecast.

## Circuits and elevation

- Circuit layouts: [Tomislav Bacinger / f1-circuits](https://github.com/bacinger/f1-circuits), MIT. Original notice and licence: [data/external/f1-circuits](data/external/f1-circuits/SOURCE.md).
- Copernicus DEM GLO-30/GLO-90: © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA. [Source and processing attribution](data/external/f1-circuits/SOURCE.md). Terrain is sampled, smoothed, and quality-gated; generated geometry is bundled locally.

## Cars, portraits and logos

- Formula F222 Ultimate: **CarlosCG31**, [original model](https://sketchfab.com/3d-models/formula-f222-ultimate-de780743451748d6b5567b1e190c7908), **CC BY 4.0**; optimized distribution via [BadCodesGG/interactive-f1](https://github.com/BadCodesGG/interactive-f1). Geometry was normalized, assembled into shared render meshes and given approximate original constructor-colour patterns. [Full model attribution and licence](frontend/public/models/ATTRIBUTION.html).
- Four historical-era bases: **3D Assets**, **CC0 1.0**; simplified generic approximations, with supplier-disclosed AI assistance. [Sources, modifications, checksums and licence](frontend/public/models/historical/README.md). The 2010–2017 range uses the disclosed 2000s approximation.
- Driver portraits and constructor marks: Wikimedia Commons, under the individual free licences recorded in [portrait manifest](frontend/src/portraits.json), [logo manifest](frontend/src/logos.json), and the [complete per-file attribution page](frontend/public/credits.html). Crops and resized derivatives retain their source licences. No blanket licence supersedes them.
- README screenshots are actual APEX browser captures. They contain the assets credited above; the [capture record](docs/assets/screenshots/capture.json) records the UI states and actual finishing order. Screenshots were compressed to WebP, without invented interface states.

Software dependencies retain their own licences. No single repository-wide software licence has been selected; the specific third-party data and asset terms above still apply.

- Fuji, Jerez and Yeongam layouts: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/). [Pinned source](https://github.com/YangKuoshih/SimgentF1/blob/74468face35a888fe183e4d49ebcb6d38acf83d4/data/osm-circuits.geojson). The app projects and resamples the mapped centreline and displays it flat.
