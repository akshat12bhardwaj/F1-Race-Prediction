"""The app's prediction: one classification + one regression model -> podium / not on podium / full standings.
Real trained models (models/algorithms); nothing is mocked."""

import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/algorithms/metadata.json").exists(), reason="algorithms not trained")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def grid(client):
    """The app's race (D-036): the default 22-car grid, 11 constructors x 2, each in its peak car."""
    return client.get("/api/driver-pool").json()["default_grid"]


@pytest.fixture(scope="module")
def grid2025(client):
    """The real 2025 field (20 drivers in their 2025 cars) for the general simulator (build_scenario) checks."""
    return client.get("/api/default-grid", params={"circuit_id": "monza"}).json()["entries"]


def with_drivers(grid, ids):
    """App grid with the first len(ids) seats taken over by these drivers (same cars, form seasons filled in)."""
    out = [{"driver_id": e["driver_id"], "constructor_id": e["constructor_id"]} for e in grid]
    for k, d in enumerate(ids):
        out[k]["driver_id"] = d
    return out


def _predict(client, entries, cls="random_forest", reg="ridge", **kw):
    return client.post("/api/race-prediction", json={"circuit_id": "monza", "entries": entries, "classification": cls,
                                                     "regression": reg, "n_sims": 1500, **kw})


def test_prediction_returns_podium_rest_and_full_standings(client, grid):
    r = _predict(client, grid)
    assert r.status_code == 200
    o = r.json()
    ids = sorted(e["driver_id"] for e in grid)
    S = o["standings"]
    assert sorted(s["driver_id"] for s in S) == ids and [s["position"] for s in S] == list(range(1, 23))
    assert o["podium"] == [s["driver_id"] for s in S[:3]] and len(set(o["podium"])) == 3
    assert o["not_podium"] == [s["driver_id"] for s in S[3:]] and not set(o["podium"]) & set(o["not_podium"])
    assert [s["expected_position"] for s in S] == sorted(s["expected_position"] for s in S)   # order = blended expectation
    for s in S:
        assert s["expected_position"] == pytest.approx((s["classification"]["expected_position"] + s["regression"]["expected_position"]) / 2)
        assert 0 <= s["podium_prob"] <= 1
    assert abs(sum(s["podium_prob"] for s in S) - 3) < 1e-9
    assert [e["driver_id"] for e in grid] == [s["driver_id"] for s in sorted(S, key=lambda s: s["grid"])]   # lineup order = grid
    m = o["models"]
    assert m["classification"]["key"] == "random_forest" and m["regression"]["key"] == "ridge"
    assert o["podium_agreement"] == (sorted(m["classification"]["podium"]) == sorted(m["regression"]["podium"]))


def test_exactly_one_model_of_each_family(client, grid):
    for cls, reg in (("ridge", "ridge"), ("knn", "knn"), ("plackett_luce", "ridge"), ("knn", "grid_baseline")):
        r = _predict(client, grid, cls, reg)
        assert r.status_code == 422 and "Choose a" in r.json()["detail"]["message"]
    assert _predict(client, grid, cls=None).status_code == 422


def test_model_choice_and_lineup_order_change_the_prediction(client, grid):
    a = _predict(client, grid, "logistic_regression", "linear_regression").json()
    b = _predict(client, grid, "svm", "decision_tree_regression").json()
    assert [s["expected_position"] for s in a["standings"]] != [s["expected_position"] for s in b["standings"]]
    back = grid[1:] + grid[:1]                     # pole sitter moved to the back of the grid
    c = _predict(client, back).json()
    first = grid[0]["driver_id"]
    p0 = next(s for s in _predict(client, grid).json()["standings"] if s["driver_id"] == first)
    p1 = next(s for s in c["standings"] if s["driver_id"] == first)
    assert p1["grid"] == 22 and p1["position"] > p0["position"] and p1["podium_prob"] < p0["podium_prob"]


def test_changing_the_constructor_changes_the_car(client, grid):
    e = [{"driver_id": x["driver_id"], "constructor_id": x["constructor_id"]} for x in grid]
    e[0]["constructor_id"], e[21]["constructor_id"] = e[21]["constructor_id"], e[0]["constructor_id"]   # swap seats
    first = grid[0]["driver_id"]
    a = next(s for s in _predict(client, grid).json()["standings"] if s["driver_id"] == first)
    b = next(s for s in _predict(client, e).json()["standings"] if s["driver_id"] == first)
    assert b["constructor_id"] == grid[21]["constructor_id"] and b["podium_prob"] < a["podium_prob"]


def test_input_validation(client, grid):
    assert _predict(client, grid[:2]).status_code == 422                                  # at least 3 drivers
    assert _predict(client, grid + grid[:1]).status_code == 422                           # duplicate driver
    r = client.post("/api/race-prediction", json={"circuit_id": "nowhere", "entries": grid, "classification": "knn", "regression": "ridge"})
    assert r.status_code == 422


# ---------------- circuit awareness (D-023) ----------------
def test_circuit_features_reach_inference_with_the_training_definition(grid2025):
    grid = grid2025
    from src import simulate
    from src.algorithms import FEATURES
    from src.features import load_features
    from src.multi_sim import _algos
    assert {"circuit_dnf_rate", "constructor_circuit_avg_grid_pct", "log_grid_x_pole_rate"} <= set(FEATURES)
    from src.algorithms import FEATURE_SETS
    assert all(b["features"] == FEATURE_SETS[b.get("feature_set", "v1")] for k, b in _algos()[0].items() if k not in ("plackett_luce", "grid_baseline"))
    a, _, _ = simulate.build_scenario("monaco", grid)
    b, _, _ = simulate.build_scenario("monza", grid)
    for c in ("circuit_dnf_rate", "circuit_pole_win_rate", "constructor_circuit_avg_grid_pct"):
        assert a[c].notna().all() or c == "constructor_circuit_avg_grid_pct"
        assert not a[c].equals(b[c]), c
    # inference value = what features.py's expanding mean would give the NEXT race at the circuit
    mf = load_features()
    per_race = mf[mf["circuit_id"] == "monza"].assign(o=lambda x: x["season"] * 100 + x["round"]).groupby("race_id").agg(o=("o", "first"), s=("target_dnf", "mean"),
                                                                       f=("circuit_dnf_rate", "first")).sort_values("o")
    n = len(per_race)
    assert a.loc[0, "circuit_dnf_rate"] != b.loc[0, "circuit_dnf_rate"]
    assert b.loc[0, "circuit_dnf_rate"] == pytest.approx((per_race["f"].iloc[-1] * (n - 1) + per_race["s"].iloc[-1]) / n)


def test_circuit_changes_predictions_where_the_model_uses_circuit_features(grid2025):
    grid = grid2025
    """Not forced: checked only for models whose fitted weights on circuit features are non-zero."""
    import numpy as np
    from src import simulate
    from src.algorithms import FEATURES
    from src.multi_sim import _algos, score_table
    bundles = _algos()[0]
    tables = {c: simulate.build_scenario(c, grid)[0] for c in ("monaco", "monza", "silverstone")}
    idx = [FEATURES.index(f) for f in ("circuit_dnf_rate", "constructor_circuit_avg_grid_pct", "log_grid_x_pole_rate")]
    for key in ("logistic_regression", "ridge", "lasso"):
        est = bundles[key]["model"].steps[-1][1]
        if not np.any(np.ravel(est.coef_)[idx]):
            continue
        s = {c: score_table(bundles[key], df) for c, df in tables.items()}
        assert not np.allclose(s["monaco"], s["monza"]) and not np.allclose(s["monza"], s["silverstone"]), key


def test_eligibility_tiers_and_diagnostics_are_kept_out_of_the_ui(client, grid):
    from src import simulate
    o = _predict(client, grid).json()
    assert o["eligibility"] == {"limited": [], "insufficient": []}          # the 2025 pool is fully supported
    ev = {s["driver_id"]: s for s in o["standings"]}
    assert all(s["eligibility"] == "supported" and s["evidence"]["imputed_skill_features"] == 0 for s in ev.values())
    assert isinstance(o["diagnostics"], list) and "warnings" not in o   # kept for developers, not for the app
    # a real driver-season with very few starts is 'limited', not dropped or invented
    snaps = simulate._data()["drivers"]
    few = snaps[(snaps["driver_career_starts"].between(1, 4)) & (snaps["season"] < 2025)].iloc[0]
    assert simulate.eligibility(few, 0)["eligibility"] == "limited"


# ---------------- lineup editor -> model inputs (the edit must reach the features, not just the labels) ----------------
def test_car_assignment_changes_the_car_features_and_keeps_the_driver(grid2025):
    grid = grid2025
    from src import simulate
    e = [dict(x) for x in grid]
    i, j = [k for k, x in enumerate(e) if x["driver_id"] in ("norris", "bortoleto")]
    e[i]["constructor_id"], e[j]["constructor_id"] = e[j]["constructor_id"], e[i]["constructor_id"]   # seat swap
    a = simulate.build_scenario("monza", grid)[0].set_index("driver_id")
    b = simulate.build_scenario("monza", e)[0].set_index("driver_id")
    car, drv = simulate.CAR_COLS, simulate.DRIVER_COLS
    assert b.loc["norris", "constructor_id"] == "sauber" and not bool(b.loc["norris", "real_pairing"])
    assert b.loc["norris", car].astype(float).equals(a.loc["bortoleto", car].astype(float))     # he gets the Sauber's numbers
    assert not b.loc["norris", car].astype(float).equals(a.loc["norris", car].astype(float))
    assert b.loc["norris", drv].astype(float).equals(a.loc["norris", drv].astype(float))        # and stays the same driver
    assert b.loc["norris", "constructor_circuit_avg_grid_pct"] == a.loc["bortoleto", "constructor_circuit_avg_grid_pct"]
    untouched = [d for d in a.index if d not in ("norris", "bortoleto")]
    assert b.loc[untouched, car + drv].astype(float).equals(a.loc[untouched, car + drv].astype(float))


def test_grid_slot_is_a_model_input_for_both_selected_models(client, grid, grid2025):
    import numpy as np
    from src import simulate
    from src.multi_sim import _algos, score_table
    df = simulate.build_scenario("monza", grid2025)[0]
    for key in ("random_forest", "ridge"):
        t = score_table(_algos()[0][key], df)
        assert t.shape == (20, 20) and np.isfinite(t).all()
        assert (t[:, 0] > t[:, -1]).all(), f"{key}: every driver scores better from pole than from P20"
    order = grid[::-1]                                       # reversed grid: last year's pole sitter starts last
    o = _predict(client, order).json()
    by = {s["driver_id"]: s for s in o["standings"]}
    assert [by[e["driver_id"]]["grid"] for e in order] == list(range(1, 23))    # entry order IS the grid the models saw
    assert o["scenario_id"] != _predict(client, grid).json()["scenario_id"]


def test_prediction_is_deterministic_and_reports_its_compute_time(client, grid):
    a, b = _predict(client, grid).json(), _predict(client, grid).json()
    strip = lambda o: {k: v for k, v in o.items() if k != "inference_ms"}
    assert strip(a) == strip(b)                              # same scenario -> same answer; nothing random replaces it
    assert a["inference_ms"] > 0 and isinstance(a["inference_ms"], float)
    assert len(set(a["podium"])) == 3 and a["podium"] == [s["driver_id"] for s in a["standings"][:3]]
    assert sorted(s["position"] for s in a["standings"]) == list(range(1, 23))


def test_2025_race_numbers_and_nationality(client, grid):
    # season-specific race numbers, not today's permanent ones: 2025 Verstappen 1, Norris 4 (the source stores 3 and 1)
    two = with_drivers(grid, ["max_verstappen", "antonelli"])      # 2025 form-season drivers in the default cars
    st = {s["driver_id"]: s["driver_number"] for s in _predict(client, two).json()["standings"]}
    assert st["max_verstappen"] == 1 and st["antonelli"] == 12 and st["hamilton"] == 44
    assert all(isinstance(n, int) for n in st.values())
    pool = {d["driver_id"]: d for d in client.get("/api/driver-pool").json()["drivers"]}
    assert pool["max_verstappen"]["season"] == 2025 and pool["max_verstappen"]["number"] == 1
    assert pool["senna"]["season"] == 1988 and pool["senna"]["number"] == 12
    assert pool["max_verstappen"]["nationality"] == "Dutch" and all(d["nationality"] for d in pool.values())
