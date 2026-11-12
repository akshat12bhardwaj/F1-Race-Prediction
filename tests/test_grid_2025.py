"""2025 reference grid, user-controlled starting grid (drag-and-drop order), car swaps and multi-model outcomes.
Real trained models only (models/algorithms, models/scenario); nothing is mocked."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/algorithms/metadata.json").exists(),
                                reason="algorithms not trained")
N = 1500


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def grid(client):
    return client.get("/api/default-grid", params={"circuit_id": "bahrain"}).json()["entries"]


@pytest.fixture(scope="module")
def ms():
    from src import multi_sim
    return multi_sim


def _sim(ms, entries, algos, circuit="bahrain", weather="dry", grid_mode="fixed"):
    return ms.simulate_algorithms(circuit, entries, algos, {"weather": weather}, N, 42, grid_mode)


def _row(out, k, d):
    return next(r for r in out["runs"][k]["results"] if r["driver_id"] == d)


def _move(entries, d, to):
    e = [x for x in entries if x["driver_id"] != d]
    e.insert(to, next(x for x in entries if x["driver_id"] == d))
    return e


# ---------------- 2025 reference field and eligibility ----------------
def test_default_grid_is_the_eligible_2025_field_in_10_teams_of_two(grid):
    assert len(grid) == 20 and len({e["driver_id"] for e in grid}) == 20
    teams = {}
    for e in grid:
        assert e["season"] == 2025 and e["constructor_season"] == 2025
        teams.setdefault(e["constructor_id"], []).append(e["driver_id"])
    assert len(teams) == 10 and all(len(v) == 2 for v in teams.values())
    # the seat each driver finished 2025 in (mid-season swaps: Tsunoda to Red Bull, Colapinto for Doohan)
    assert sorted(teams["red_bull"]) == ["max_verstappen", "tsunoda"] and sorted(teams["rb"]) == ["hadjar", "lawson"]
    assert sorted(teams["alpine"]) == ["colapinto", "gasly"]


def test_2025_reference_audit_still_excludes_thin_evidence():
    from src import simulate
    p = simulate.reference_pool()                      # the 2025 audit (docs/DRIVER_DATA_AUDIT.md); default grid = its field
    assert p["reference_season"] == 2025 and [d["driver_id"] for d in p["excluded"]] == ["doohan"]
    assert all(d["career_starts"] >= 10 and d["season_starts"] >= 5 and d["imputed_skill_features"] == 0 for d in p["eligible"])


def test_prediction_rejects_drivers_outside_the_pool_and_invalid_grids(client):
    """The app's race (D-036): 22 pool drivers, 11 constructors x 2, form seasons and peak cars only."""
    def post(entries):
        return client.post("/api/race-prediction", json={"circuit_id": "bahrain", "entries": entries,
                                                         "classification": "random_forest", "regression": "ridge", "n_sims": 200})
    app = [{"driver_id": e["driver_id"], "constructor_id": e["constructor_id"]} for e in client.get("/api/driver-pool").json()["default_grid"]]
    msg = lambda r: r.json()["detail"]["message"]
    r = post([{**app[0], "driver_id": "doohan"}] + app[1:])
    assert r.status_code == 422 and "doohan" in msg(r)
    r = post([{**e, "season": 2026} for e in app])
    assert r.status_code == 422 and "form season" in msg(r)
    r = post([{**e, "constructor_id": app[0]["constructor_id"]} if i == 2 else e for i, e in enumerate(app)])
    assert r.status_code == 422 and "11 constructors with 2 drivers each" in msg(r)
    assert post(app + app[:1]).status_code == 422
    assert post(app[:20]).status_code == 422
    assert post(app).status_code == 200


# ---------------- every model: valid podium and complete ranking ----------------
def test_every_model_returns_valid_podium_ranking_and_distinct_outcomes(ms, grid):
    out = _sim(ms, grid, "all")
    assert len(out["runs"]) == 18
    ids = sorted(e["driver_id"] for e in grid)
    for k, run in out["runs"].items():
        res = run["results"]
        assert sorted(r["driver_id"] for r in res) == ids, k
        assert sorted(r["pred_rank"] for r in res) == list(range(1, 21)), k
        assert len(set(run["podium"])) == 3 and run["podium"] == [r["driver_id"] for r in res[:3]], k
        assert abs(sum(r["win_prob"] for r in res) - 1) < 1e-9 and abs(sum(r["podium_prob"] for r in res) - 3) < 1e-9
        best = max(res, key=lambda r: r["win_prob"])
        assert run["most_likely_winner"]["win_prob"] == best["win_prob"], k   # mathematically the highest win %
        assert len(set(run["most_likely_podium"]["driver_ids"])) == 3 and 0 < run["most_likely_podium"]["prob"] <= 1
        assert sorted(run["sample_race"]) == ids and res[0]["factors"]
    # same scenario, different models -> genuinely different predictions (not hardcoded)
    assert len({tuple(r["podium"]) for r in out["runs"].values()}) > 1


def test_comparison_uses_identical_inputs(ms, grid):
    both = _sim(ms, grid, ["knn", "ridge"])
    for k in ("knn", "ridge"):
        alone = _sim(ms, grid, [k])
        assert alone["scenario_id"] == both["scenario_id"]
        assert [(r["driver_id"], r["win_prob"], r["est_grid"]) for r in alone["runs"][k]["results"]] == \
               [(r["driver_id"], r["win_prob"], r["est_grid"]) for r in both["runs"][k]["results"]]


# ---------------- user customisation reaches the features ----------------
def test_fixed_grid_order_is_the_starting_grid_and_drag_changes_it(ms, grid):
    out = _sim(ms, grid, ["plackett_luce"])
    est = {r["driver_id"]: (r["est_grid"], r["pole_prob"]) for r in out["runs"]["plackett_luce"]["results"]}
    assert all(est[e["driver_id"]] == (i + 1, 1.0 if i == 0 else 0.0) for i, e in enumerate(grid))
    keys = ["plackett_luce", "logistic_regression", "decision_tree", "knn", "random_forest", "ridge", "svm"]
    front, back = _sim(ms, _move(grid, "hamilton", 0), keys), _sim(ms, _move(grid, "hamilton", 19), keys)
    for k in keys:
        f, b = _row(front, k, "hamilton"), _row(back, k, "hamilton")
        assert f["est_grid"] == 1 and b["est_grid"] == 20
        assert f["expected_position"] < b["expected_position"] - 2 and f["win_prob"] > b["win_prob"], k
    assert front["scenario_id"] != back["scenario_id"]


def test_swapping_cars_changes_car_features_and_prediction(ms, grid):
    e = [dict(x) for x in grid]
    i, j = [k for k, x in enumerate(e) if x["driver_id"] in ("norris", "bortoleto")]
    for f in ("constructor_id", "constructor_season"):
        e[i][f], e[j][f] = e[j][f], e[i][f]
    a, b = _sim(ms, grid, ["plackett_luce"]), _sim(ms, e, ["plackett_luce"])
    na, nb = _row(a, "plackett_luce", "norris"), _row(b, "plackett_luce", "norris")
    assert nb["constructor_id"] == "sauber" and nb["est_grid"] == na["est_grid"]          # same slot, other car
    assert nb["inputs"]["constructor_points_share_before_race"] < na["inputs"]["constructor_points_share_before_race"]
    assert nb["win_prob"] < na["win_prob"] and _row(b, "plackett_luce", "bortoleto")["win_prob"] > _row(a, "plackett_luce", "bortoleto")["win_prob"]


def test_circuit_change_keeps_driver_and_car_assignments(grid):
    from src.simulate import build_scenario
    cols = ["driver_id", "season", "constructor_id", "constructor_season"]
    a, _, _ = build_scenario("bahrain", grid)
    b, _, _ = build_scenario("monaco", grid)
    assert a[cols].equals(b[cols]) and (a["circuit_pole_win_rate"] != b["circuit_pole_win_rate"]).all()


def test_qualifying_mode_is_shared_by_all_models_and_independent_of_the_selected_model(ms, grid):
    out = _sim(ms, grid, ["plackett_luce", "decision_tree"], grid_mode="qualifying")
    g = {k: [(r["driver_id"], r["est_grid"], r["pole_prob"]) for r in sorted(run["results"], key=lambda r: r["driver_id"])]
         for k, run in out["runs"].items()}
    assert g["plackett_luce"] == g["decision_tree"] and "qualifying model" in out["qualifying"]
    assert any(p not in (0.0, 1.0) for _, _, p in g["plackett_luce"])      # genuinely sampled, not fixed
