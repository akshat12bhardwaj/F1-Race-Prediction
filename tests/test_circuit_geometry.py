"""Circuit elevation processing (src/circuit_geometry.py): smoothing, grades and the quality gate. No network."""

import json

import numpy as np
import pytest

from src import circuit_geometry as cg
from src.config import PROJECT_ROOT


def test_closed_loop_smoothing_has_no_seam_and_keeps_level_ground_level():
    assert np.allclose(cg.smooth_closed(np.full(320, 412.0), 15), 412.0)
    hill = 40 * np.sin(np.linspace(0, 2 * np.pi, 320, endpoint=False))       # one smooth climb and descent
    noisy = hill + np.random.default_rng(0).normal(0, 3, 320)                # DEM surface noise (trees, buildings)
    y = cg.smooth_closed(noisy, 15)
    assert len(y) == 320 and np.std(y - hill) < 1.5                          # noise removed, shape kept
    assert abs(y[0] - y[-1]) < 3                                             # wraps across the start line


def test_grade_of_a_known_ramp():
    y = np.concatenate([np.linspace(0, 20, 160), np.linspace(20, 0, 160)])  # 20 m over 160 x 10 m = 1.25%
    assert cg.profile_stats(y, 10) == pytest.approx(20 / 1590, rel=0.05)


def test_dem_tile_names_follow_the_copernicus_grid():
    assert cg._tile_name(50.43, 5.97) == "Copernicus_DSM_COG_10_N50_00_E005_00_DEM"
    assert cg._tile_name(-23.70, -46.69) == "Copernicus_DSM_COG_10_S24_00_W047_00_DEM"   # Interlagos


def test_built_circuits_label_elevation_honestly():
    out = PROJECT_ROOT / "frontend/public/circuits.json"
    if not out.exists():
        pytest.skip("run python -m src.circuit_geometry")
    d = json.loads(out.read_text(encoding="utf-8"))
    assert "Copernicus" in d["elevation_source"]
    for cid, c in d["circuits"].items():
        e = c["elevation"]
        if e["quality"] == "measured":
            assert len(c["y"]) == len(c["x"]) and min(c["y"]) == 0 and e["max_grade"] <= cg.GRADE_MAX, cid
            assert e["noise_m"] <= cg.NOISE_MAX, cid
        else:
            assert "y" not in c and "terrain" not in c, cid                   # flat, never invented


def test_osm_layouts_are_scaled_flat_and_attributed():
    d = json.loads((PROJECT_ROOT / "frontend/public/circuits-osm.json").read_text(encoding="utf-8"))
    assert d["license"] == "ODbL 1.0"
    assert set(d["circuits"]) == {"fuji", "jerez", "yeongam"}
    for c in d["circuits"].values():
        assert c["source"] == "OpenStreetMap contributors"
        assert len(c["x"]) == len(c["z"]) == cg.N_POINTS
        assert c["traced_m"] == pytest.approx(c["length_m"], rel=.015)
        assert "y" not in c and "terrain" not in c
