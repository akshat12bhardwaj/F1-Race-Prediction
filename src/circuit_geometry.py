"""
Circuit geometry for the 3D track view (visualisation only; never a model feature).

Layout: data/external/f1-circuits/f1-circuits.geojson (bacinger/f1-circuits, MIT, see SOURCE.md).
Elevation: Copernicus DEM GLO-30 (30 m), public tiles on AWS Open Data (registry.opendata.aws/copernicus-dem),
(c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA.
Each 1x1 degree tile is downloaded to the system temp folder, sampled (bilinear) and deleted; the samples are cached
in data/raw/dem/<circuit>.json, so a rebuild never re-downloads.

Each circuit is projected from lon/lat to local metres (equirectangular around its centre) and resampled to evenly
spaced points. The DEM is sampled at those points and on a terrain grid around the circuit. The DEM is a
30 m surface model (it includes trees and buildings), so the centreline profile is smoothed along the track
(Gaussian, sigma 60 m) and every circuit is graded:
    measured     residual noise <= NOISE_MAX and steepest 100 m grade <= GRADE_MAX: real terrain profile
    approximate  noisier or implausibly steep (buildings, bridges, cuttings): shown flat and labelled
Nothing is invented: a circuit either gets its DEM profile or a flat, labelled surface.

Run: python -m src.circuit_geometry
"""

import json
import math
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "data/external/f1-circuits/f1-circuits.geojson"
OUT = PROJECT_ROOT / "frontend/public/circuits.json"
CACHE = PROJECT_ROOT / "data/raw/dem"
TILES = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
TILES_90 = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"
N_POINTS, GRID_N, GRID_SUB = 320, 40, 5          # terrain node = mean of GRID_SUB^2 DEM samples (softens tree cover)
SIGMA_M, NOISE_MAX, GRADE_MAX = 60.0, 2.5, 0.20      # real F1 tracks peak near 18% (Eau Rouge/Raidillon)
R_EARTH = 6_371_000.0
# Jolpica circuit_id -> geojson id (current/recent layouts; historic layouts at the same venue may differ)
MATCH = {"albert_park": "au-1953", "bahrain": "bh-2002", "shanghai": "cn-2004", "catalunya": "es-1991", "monaco": "mc-1929",
         "villeneuve": "ca-1978", "ricard": "fr-1969", "red_bull_ring": "at-1969", "silverstone": "gb-1948",
         "hockenheimring": "de-1932", "hungaroring": "hu-1986", "spa": "be-1925", "monza": "it-1922", "marina_bay": "sg-2008",
         "sochi": "ru-2014", "suzuka": "jp-1962", "americas": "us-2012", "rodriguez": "mx-1962", "interlagos": "br-1940",
         "yas_marina": "ae-2009", "imola": "it-1953", "nurburgring": "de-1927", "portimao": "pt-2008", "mugello": "it-1914",
         "sepang": "my-1999", "istanbul": "tr-2005", "zandvoort": "nl-1948", "magny_cours": "fr-1960", "estoril": "pt-1972",
         "jacarepagua": "br-1977", "jeddah": "sa-2021", "miami": "us-2022", "losail": "qa-2004", "baku": "az-2016",
         "vegas": "us-2023", "indianapolis": "us-1909", "galvez": "ar-1952", "kyalami": "za-1961", "watkins_glen": "us-1956"}


def centreline(coords, n=N_POINTS):
    """Evenly spaced points: local x/z metres (north = -z, three.js convention), their lon/lat, and the traced length."""
    lon, lat = np.array(coords, dtype=float).T
    if np.hypot(lon[0] - lon[-1], lat[0] - lat[-1]) > 1e-7:          # close the loop
        lon, lat = np.append(lon, lon[0]), np.append(lat, lat[0])
    lon0, lat0 = lon.mean(), lat.mean()
    x = np.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    z = -np.radians(lat - lat0) * R_EARTH
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
    t = np.linspace(0, s[-1], n, endpoint=False)
    f = lambda v: np.interp(t, s, v)
    return f(x), f(z), f(lon), f(lat), float(s[-1]), (lon0, lat0)


def _tile_name(lat, lon):
    la, lo = math.floor(lat), math.floor(lon)
    return f"Copernicus_DSM_COG_10_{'N' if la >= 0 else 'S'}{abs(la):02d}_00_{'E' if lo >= 0 else 'W'}{abs(lo):03d}_00_DEM"


def sample_dem(lats, lons):
    """Bilinear DEM heights (m) at many points; each needed tile is fetched once, then deleted."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    out = np.full(len(lats), np.nan)
    groups = defaultdict(list)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        groups[_tile_name(la, lo)].append(i)
    for name, idx in groups.items():
        path = f"{tempfile.gettempdir()}/{name}.tif"
        try:
            urllib.request.urlretrieve(TILES.format(n=name), path)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            try:                           # a few countries' 30 m tiles are withheld (e.g. Azerbaijan): use GLO-90
                n90 = name.replace("_COG_10_", "_COG_30_")
                urllib.request.urlretrieve(TILES_90.format(n=n90), path)
            except urllib.error.HTTPError as e2:
                if e2.code == 404:         # open water: no tile at all; leave NaN, never guess a height
                    continue
                raise
        a = np.array(Image.open(path), dtype=np.float64)         # row 0 = north edge, col 0 = west edge
        h, w = a.shape
        idx = np.array(idx)
        r = (math.floor(lats[idx[0]]) + 1 - lats[idx]) * h
        c = (lons[idx] - math.floor(lons[idx[0]])) * w
        r0, c0 = np.clip(np.floor(r).astype(int), 0, h - 2), np.clip(np.floor(c).astype(int), 0, w - 2)
        fr, fc = np.clip(r - r0, 0, 1), np.clip(c - c0, 0, 1)
        out[idx] = (a[r0, c0] * (1 - fr) * (1 - fc) + a[r0, c0 + 1] * (1 - fr) * fc
                    + a[r0 + 1, c0] * fr * (1 - fc) + a[r0 + 1, c0 + 1] * fr * fc)
        del a
        Path(path).unlink()
    return out


def elevation(cid, lon, lat, grid_lon, grid_lat):
    """Raw DEM metres along the centreline and on the terrain sub-grid (cached per circuit)."""
    path = CACHE / f"{cid}.json"
    if path.exists():
        return json.loads(path.read_text())
    v = sample_dem(np.concatenate([lat, grid_lat]), np.concatenate([lon, grid_lon]))
    raw = {"source": "Copernicus DEM GLO-30 (AWS Open Data tiles)", "retrieved": time.strftime("%Y-%m-%d"),
           "line": [None if np.isnan(a) else round(float(a), 2) for a in v[:len(lat)]],
           "grid": [None if np.isnan(a) else round(float(a), 2) for a in v[len(lat):]]}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    return raw


def smooth_closed(v, step_m, sigma_m=SIGMA_M):
    """Gaussian smoothing along a closed loop (wraps at the start line, so there is no seam)."""
    k = int(math.ceil(3 * sigma_m / step_m))
    w = np.exp(-0.5 * (np.arange(-k, k + 1) * step_m / sigma_m) ** 2)
    w /= w.sum()
    return np.convolve(np.concatenate([v[-k:], v, v[:k]]), w, mode="valid")


def profile_stats(y, step_m):
    """Range, residual noise is computed by the caller; here: steepest grade over ~100 m windows (both directions)."""
    k = max(1, int(round(100 / step_m)))
    return float(np.max(np.abs(np.roll(y, -k) - y)) / (k * step_m))


def build(cid, feature):
    x, z, lon, lat, traced, (lon0, lat0) = centreline(feature["geometry"]["coordinates"])
    step = traced / N_POINTS
    m = 0.25 * max(np.ptp(x), np.ptp(z))                              # terrain margin around the circuit
    gx = np.linspace(x.min() - m, x.max() + m, GRID_N)
    gz = np.linspace(z.min() - m, z.max() + m, GRID_N)
    sub = (np.arange(GRID_SUB) - (GRID_SUB - 1) / 2) / GRID_SUB          # sub-samples spread over one cell
    GX = (gx[None, :, None, None] + sub[None, None, None, :] * (gx[1] - gx[0])) + 0 * gz[:, None, None, None]
    GZ = (gz[:, None, None, None] + sub[None, None, :, None] * (gz[1] - gz[0])) + 0 * gx[None, :, None, None]
    GX, GZ = np.broadcast_arrays(GX, GZ)                              # [row z, col x, sub z, sub x]
    glon = lon0 + np.degrees(GX.ravel() / (R_EARTH * math.cos(math.radians(lat0))))
    glat = lat0 - np.degrees(GZ.ravel() / R_EARTH)
    raw = elevation(cid, lon, lat, glon, glat)
    line = np.array([np.nan if v is None else v for v in raw["line"]], dtype=float)
    if np.isnan(line).any():               # part of the track has no DEM coverage: draw it flat, labelled
        return {"name": feature["properties"]["Name"], "length_m": feature["properties"].get("length"), "traced_m": round(traced),
                "x": np.round(x, 1).tolist(), "z": np.round(z, 1).tolist(),
                "elevation": {"quality": "approximate", "reason": "no DEM coverage on part of the track"}}
    grid = np.array([np.nan if v is None else v for v in raw["grid"]], dtype=float)
    grid[np.isnan(grid)] = np.nanmin(grid)  # open water: no tile, shown at the lowest land sample nearby
    y = smooth_closed(line, step)
    noise = float(np.std(line - y))
    grade = profile_stats(y, step)
    quality = "measured" if noise <= NOISE_MAX and grade <= GRADE_MAX else "approximate"
    base = float(y.min())
    out = {"name": feature["properties"]["Name"], "length_m": feature["properties"].get("length"), "traced_m": round(traced),
           "x": np.round(x, 1).tolist(), "z": np.round(z, 1).tolist(),
           "elevation": {"quality": quality, "range_m": round(float(np.ptp(y)), 1), "max_grade": round(grade, 3),
                         "noise_m": round(noise, 2), "base_m": round(base, 1)}}
    if quality == "measured":
        out["y"] = np.round(y - base, 2).tolist()
        out["terrain"] = {"n": GRID_N, "x0": round(float(gx[0]), 1), "z0": round(float(gz[0]), 1),
                          "dx": round(float(gx[1] - gx[0]), 2), "dz": round(float(gz[1] - gz[0]), 2),
                          "h": np.round(grid.reshape(GRID_N * GRID_N, -1).mean(1) - base, 1).tolist()}
    return out


def main():
    g = json.loads(SRC.read_text(encoding="utf-8"))
    by_id = {f["properties"]["id"]: f for f in g["features"]}
    out = {cid: build(cid, by_id[gid]) for cid, gid in MATCH.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "bacinger/f1-circuits (MIT), current layouts",
                               "elevation_source": "Copernicus DEM GLO-30 (© DLR e.V. 2010-2014 and © Airbus Defence and "
                                                   "Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA); "
                                                   "smoothed along the track",
                               "circuits": out}, separators=(",", ":")))
    for cid, v in sorted(out.items(), key=lambda kv: -kv[1]["elevation"].get("range_m", -1)):
        e = v["elevation"]
        if "range_m" not in e:
            print(f"{cid:15s} {e['quality']:11s} {e['reason']}")
            continue
        print(f"{cid:15s} {e['quality']:11s} range {e['range_m']:6.1f} m  grade {100 * e['max_grade']:5.1f}%  noise {e['noise_m']:.2f} m")
    bad = {k: (v["length_m"], v["traced_m"]) for k, v in out.items() if v["length_m"] and abs(v["traced_m"] / v["length_m"] - 1) > 0.05}
    print(f"{len(out)} circuits -> {OUT} ({OUT.stat().st_size // 1024} KB); traced length off by >5%: {bad or 'none'}")


if __name__ == "__main__":
    main()
