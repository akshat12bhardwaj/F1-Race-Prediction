"""D-036: the app's race is 22 drivers in 11 constructors of two; every constructor races its peak car-season.
Real data and trained models; nothing is mocked."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/algorithms/metadata.json").exists(), reason="algorithms not trained")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def S():
    from src import simulate
    return simulate


@pytest.fixture(scope="module")
def grid(S):
    return [dict(e) for e in S.peak_default_grid()]


def _post(client, entries, circuit="monza", cls="random_forest", reg="ridge"):
    return client.post("/api/race-prediction", json={"circuit_id": circuit, "entries": entries, "classification": cls,
                                                     "regression": reg, "n_sims": 1500})


def _season_table(S):
    h = S._data()["hist"]
    s = h.groupby(["season", "constructor_id"]).agg(grid_pct=("grid_pct", "mean"), finish_pct=("finish_pct", "mean"),
                                                    dnf=("target_dnf", "mean"), races=("race_id", "nunique")).reset_index()
    s["dnf_vs_field"] = s["dnf"] - s["season"].map(h.groupby("season")["target_dnf"].mean())
    s["score"] = -(0.4 * s["grid_pct"] + 0.4 * s["finish_pct"] + 0.2 * s["dnf_vs_field"])
    return s


def test_one_peak_car_per_eligible_constructor_recomputed_independently(S):
    peaks = S.peak_cars()
    pool_cars = {(c["constructor_id"], c["season"]) for c in S.history_pool()["cars"]}
    ids = {c for c, _ in pool_cars}
    assert {p["constructor_id"] for p in peaks} == ids and len(peaks) == len(ids) == 65
    s = _season_table(S)
    s = s[[k in pool_cars for k in zip(s["constructor_id"], s["season"])]]
    for p in peaks:
        g = s[s["constructor_id"] == p["constructor_id"]]
        cand = g[g["races"] >= S.PEAK_MIN_RACES] if (g["races"] >= S.PEAK_MIN_RACES).any() else g
        assert p["season"] == int(cand.sort_values(["score", "season"], ascending=False).iloc[0]["season"]), p
        assert p["low_evidence"] == (not (g["races"] >= S.PEAK_MIN_RACES).any())
        assert p["peak_score"] == pytest.approx(float(cand["score"].max()), abs=1e-4)
    assert S.peak_cars() == peaks                                                    # deterministic (cached, no randomness)


def test_peak_metric_is_era_relative_not_wins_or_lap_times(S):
    h = S._data()["hist"]
    # within-race percentiles: every season's field centres on the middle (0 = pole/win, 1 = last), whatever its size
    m = h.groupby("season")[["grid_pct", "finish_pct"]].mean()
    assert m.min().min() > 0.45 and m.max().max() < 0.56
    # reliability relative to the season's field: the 1980s' ~40% DNF rate is not held against an 80s car
    assert h[h["season"] < 1990]["target_dnf"].mean() > 2 * h[h["season"] > 2015]["target_dnf"].mean()
    per = h.assign(rel=h["target_dnf"] - h["season"].map(h.groupby("season")["target_dnf"].mean())).groupby("season")["rel"].mean()
    assert per.abs().max() < 1e-9
    # widely recognised dominant cars come out as their constructor's peak
    peak = {p["constructor_id"]: p["season"] for p in S.peak_cars()}
    assert peak["mclaren"] == 1988 and peak["ferrari"] == 2002 and peak["brawn"] == 2009
    assert peak["williams"] in (1992, 1993) and peak["mercedes"] in range(2014, 2021) and peak["red_bull"] in (2011, 2013, 2023)


def test_peak_is_one_coherent_real_car_season(S, grid):
    """The car features used are exactly one real snapshot row (never a per-feature best across seasons)."""
    df = S.build_scenario("monza", S.check_app_grid(grid))[0]
    teams = S._data()["teams"].set_index(["constructor_id", "season"])
    peak = {p["constructor_id"]: p["season"] for p in S.peak_cars()}
    for _, r in df.iterrows():
        assert r["constructor_season"] == peak[r["constructor_id"]]
        assert r[S.CAR_COLS].astype(float).tolist() == pytest.approx(
            teams.loc[(r["constructor_id"], r["constructor_season"]), S.CAR_COLS].astype(float).tolist(), nan_ok=True)


def test_default_grid_is_22_real_pairings_in_11_peak_cars(S, grid):
    assert len(grid) == 22 and len({e["driver_id"] for e in grid}) == 22
    n = pd.Series([e["constructor_id"] for e in grid]).value_counts()
    assert len(n) == 11 and (n == 2).all()
    peak = {p["constructor_id"]: p for p in S.peak_cars()}
    driven = S._data()["driven"]
    form = {d["driver_id"]: d["season"] for d in S.history_pool()["drivers"]}
    for e in grid:
        assert e["constructor_season"] == peak[e["constructor_id"]]["season"] and e["season"] == form[e["driver_id"]]
        assert (e["driver_id"], e["constructor_id"], e["constructor_season"]) in driven, e   # raced that very car
    scores = [peak[e["constructor_id"]]["peak_score"] for e in grid[::2]]
    assert scores == sorted(scores, reverse=True)
    assert S.check_app_grid(grid)


@pytest.mark.parametrize("change, msg", [
    (lambda g: g[:21], "exactly 22"),
    (lambda g: g + [dict(g[0], driver_id="alesi")], "exactly 22"),
    (lambda g: [dict(e, driver_id=g[0]["driver_id"]) if i == 1 else e for i, e in enumerate(g)], "Duplicate"),
    (lambda g: [dict(e, constructor_id=g[0]["constructor_id"], constructor_season=None) if i == 21 else e for i, e in enumerate(g)], "11 constructors"),
    (lambda g: [dict(e, constructor_id="minardi", constructor_season=None) if i == 21 else e for i, e in enumerate(g)], "11 constructors"),
    (lambda g: [dict(e, constructor_season=e["constructor_season"] - 1) if i == 0 else e for i, e in enumerate(g)], "peak"),
    (lambda g: [dict(e, season=1999) if i == 0 else e for i, e in enumerate(g)], "form season"),
    (lambda g: [dict(e, constructor_id="nowhere") if i == 0 else e for i, e in enumerate(g)], "eligible constructors"),
    (lambda g: [dict(e, driver_id="fangio") if i == 0 else e for i, e in enumerate(g)], "pool"),
])
def test_app_grid_rejects_anything_but_22_by_11_by_2(S, grid, change, msg):
    with pytest.raises(S.ScenarioError, match=msg):
        S.check_app_grid(change([dict(e) for e in grid]))


def test_constructor_only_input_gets_the_peak_and_driver_features_stay_their_own(S, grid):
    """The user sends just driver + constructor; seasons are filled in. Senna in the peak Mercedes keeps his own
    form-season driver features; only the car terms come from Mercedes."""
    g = [{"driver_id": e["driver_id"], "constructor_id": e["constructor_id"]} for e in grid]
    i = next(k for k, e in enumerate(g) if e["driver_id"] != "senna" and e["constructor_id"] != "mclaren")
    j = next(k for k, e in enumerate(g) if e["driver_id"] == "senna")
    g[i]["constructor_id"], g[j]["constructor_id"] = g[j]["constructor_id"], g[i]["constructor_id"]   # seat swap
    df = S.build_scenario("monza", S.check_app_grid(g))[0].set_index("driver_id")
    snaps = S._data()["drivers"].set_index(["driver_id", "season"])
    form = {d["driver_id"]: d["season"] for d in S.history_pool()["drivers"]}
    assert df.loc["senna", "season"] == form["senna"]
    assert df.loc["senna", S.DRIVER_COLS].astype(float).tolist() == pytest.approx(
        snaps.loc[("senna", form["senna"]), S.DRIVER_COLS].astype(float).tolist())
    assert df.loc["senna", "constructor_id"] == grid[i]["constructor_id"]


def test_api_predicts_a_full_22_car_race_and_constructor_changes_reach_the_models(client, grid):
    pool = client.get("/api/driver-pool").json()
    assert len(pool["teams"]) == 65 and pool["default_grid"] == grid and pool["grid"]["drivers"] == 22
    names = [t["constructor_name"] for t in pool["teams"]]
    assert names == sorted(names, key=str.casefold)
    a = _post(client, grid).json()
    assert len(a["standings"]) == 22
    assert sorted(r["position"] for r in a["standings"]) == list(range(1, 23)) and len(a["podium"]) == 3
    assert {r["driver_id"] for r in a["standings"]} == {e["driver_id"] for e in grid}
    # swap the seats of the first and last constructors: car features, and with them the prediction, change
    g = [dict(e) for e in grid]
    for k in (0, 1):
        g[k]["constructor_id"], g[20 + k]["constructor_id"] = grid[20 + k]["constructor_id"], grid[k]["constructor_id"]
        g[k]["constructor_season"], g[20 + k]["constructor_season"] = grid[20 + k]["constructor_season"], grid[k]["constructor_season"]
    b = _post(client, g).json()
    pa = {r["driver_id"]: r["podium_prob"] for r in a["standings"]}
    pb = {r["driver_id"]: r["podium_prob"] for r in b["standings"]}
    assert pb[grid[0]["driver_id"]] < pa[grid[0]["driver_id"]]          # pole driver moved to the weakest selected car


def test_api_rejects_invalid_grids_and_newer_circuits_take_22(client, grid):
    r = _post(client, grid[:20])
    assert r.status_code == 422 and "exactly 22" in r.json()["detail"]["message"]
    r = _post(client, grid, circuit="miami")                                # Miami only ever held 20 cars in the data
    assert r.status_code == 200 and len(r.json()["standings"]) == 22


@pytest.mark.parametrize("weather", ["dry", "wet"])
def test_all_installed_model_families_support_peak_22_driver_features(client, grid, weather):
    """Exercise every selectable artifact on the full cross-era field, including both weather schemas."""
    from src.multi_sim import catalog
    models = catalog()
    cls = [m["key"] for m in models if m["family"] == "classification"]
    reg = [m["key"] for m in models if m["family"] == "regression"]
    assert len(cls) == 10 and len(reg) == 6
    for i, classifier in enumerate(cls):
        response = client.post("/api/race-prediction", json={
            "circuit_id": "monza", "entries": grid, "classification": classifier,
            "regression": reg[i % len(reg)], "weather": weather, "n_sims": 100,
        })
        assert response.status_code == 200, (classifier, response.text)
        out = response.json()
        assert [r["position"] for r in out["standings"]] == list(range(1, 23))
        assert out["podium"] == [r["driver_id"] for r in out["standings"][:3]]
        assert sum(r["win_prob"] for r in out["standings"]) == pytest.approx(1)
        assert sum(r["podium_prob"] for r in out["standings"]) == pytest.approx(3)


def test_requested_hypothetical_pairings_keep_independent_driver_and_car_features(S, grid):
    targets = {0: "michael_schumacher", 2: "hamilton", 4: "senna", 20: "max_verstappen"}
    remaining = iter(e["driver_id"] for e in grid if e["driver_id"] not in targets.values())
    entries = [{"driver_id": targets[i] if i in targets else next(remaining), "constructor_id": e["constructor_id"]}
               for i, e in enumerate(grid)]
    for i in (20, 21):
        entries[i]["constructor_id"] = "williams"
    rows = S.build_scenario("monza", S.check_app_grid(entries))[0].set_index("driver_id")
    drivers = S._data()["drivers"].set_index(["driver_id", "season"])
    cars = S._data()["teams"].set_index(["constructor_id", "season"])
    for driver, team in [("senna", "mercedes"), ("michael_schumacher", "mclaren"),
                         ("hamilton", "ferrari"), ("max_verstappen", "williams")]:
        row = rows.loc[driver]
        assert row["constructor_id"] == team
        assert row[S.DRIVER_COLS].astype(float).tolist() == pytest.approx(
            drivers.loc[(driver, row["season"]), S.DRIVER_COLS].astype(float).tolist())
        assert row[S.CAR_COLS].astype(float).tolist() == pytest.approx(
            cars.loc[(team, row["constructor_season"]), S.CAR_COLS].astype(float).tolist())
