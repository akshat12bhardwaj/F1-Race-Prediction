"""D-032: one historical pool (1981-2025) of drivers and car-seasons, cross-era scenarios, rejection, environment.
Real trained models; nothing is mocked."""

import math

import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/algorithms/metadata.json").exists(), reason="algorithms not trained")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def pool(client):
    return client.get("/api/driver-pool").json()


@pytest.fixture(scope="module")
def grid(client):
    return client.get("/api/default-grid", params={"circuit_id": "monza"}).json()["entries"]


def _post(client, entries, weather="dry", cls="random_forest", reg="ridge"):
    return client.post("/api/race-prediction", json={"circuit_id": "monza", "entries": entries, "weather": weather,
                                                     "classification": cls, "regression": reg, "n_sims": 1500})


def test_pool_spans_every_era_of_the_training_data_under_one_rule(pool):
    from src import simulate
    lo, hi = pool["model_era"]
    assert (lo, hi) == (1981, 2025) and pool["policy"]["model_era_start"] == 1981
    seasons = {d["season"] for d in pool["drivers"]}
    assert min(seasons) < 1990 and max(seasons) == 2025 and len(pool["drivers"]) >= 150
    by = {d["driver_id"]: d for d in pool["drivers"]}
    for legend in ("senna", "prost", "michael_schumacher", "hamilton", "max_verstappen", "alonso", "piquet", "mansell"):
        assert legend in by, legend
    assert "fangio" not in by and "doohan" not in by                      # before the models' data / too little evidence
    assert pool["excluded"]["before_model_era"] > 0 and pool["excluded"]["thin_evidence"] > 0
    snaps = simulate._data()["drivers"].set_index(["driver_id", "season"])
    for d in pool["drivers"]:                                             # same rule for everyone: supported, in the era
        assert set(d["seasons"]) <= set(range(lo, hi + 1)) and d["season"] in d["seasons"]
        for y in d["seasons"]:
            assert simulate.eligibility(snaps.loc[(d["driver_id"], y)], 0)["eligibility"] == "supported"


def test_car_pool_has_complete_season_specific_cars(pool):
    from src import simulate
    cars = {(c["constructor_id"], c["season"]) for c in pool["cars"]}
    assert len(cars) == len(pool["cars"]) >= 400
    for k in [("mercedes", 2020), ("mclaren", 2025), ("ferrari", 2004), ("red_bull", 2023), ("mclaren", 1988), ("williams", 1992)]:
        assert k in cars, k
    assert all(1981 <= y <= 2025 for _, y in cars)
    t = simulate._data()["teams"].set_index(["constructor_id", "season"])
    races = simulate._data()["team_race"].groupby(["constructor_id", "season"]).size()
    for k in cars:
        assert t.loc[k, simulate.MODEL_CAR_COLS].notna().all() and races[k] >= simulate.MIN_CAR_RACES, k


def test_cross_era_driver_and_car_features_come_from_separate_sources(grid):
    """Senna (his 1988 form) in the 2020 Mercedes: his driver features are his own 1988 ones, the car features are
    exactly the 2020 Mercedes ones (the car Hamilton drove), never a relabelled copy of his McLaren."""
    from src import simulate
    e = [dict(x) for x in grid if x["driver_id"] != "hamilton"][2:]
    e[0] = {"driver_id": "senna", "constructor_id": "mercedes", "constructor_season": 2020}
    e[1] = {"driver_id": "hamilton", "season": 2020, "constructor_id": "mercedes", "constructor_season": 2020}
    df = simulate.build_scenario("monza", simulate.check_entries(e))[0].set_index("driver_id")
    snaps = simulate._data()["drivers"].set_index(["driver_id", "season"])
    teams = simulate._data()["teams"].set_index(["constructor_id", "season"])
    car, drv = simulate.CAR_COLS, simulate.DRIVER_COLS
    assert df.loc["senna", "season"] == 1988 and df.loc["senna", "constructor_season"] == 2020 and not df.loc["senna", "real_pairing"]
    assert df.loc["senna", car].astype(float).equals(df.loc["hamilton", car].astype(float))
    assert df.loc["senna", car].astype(float).tolist() == pytest.approx(teams.loc[("mercedes", 2020), car].astype(float).tolist(), nan_ok=True)
    assert df.loc["senna", drv].astype(float).tolist() == pytest.approx(snaps.loc[("senna", 1988), drv].astype(float).tolist())
    mcl88 = teams.loc[("mclaren", 1988), car].astype(float)
    assert not df.loc["senna", car].astype(float).equals(mcl88)


def test_cross_era_inputs_are_normalised_within_their_own_season(pool):
    """Car inputs are season-relative rates and percentiles (0-1); driver skill is one connected teammate-network scale.
    So a 1988 car and a 2020 car land on the same scale, with no raw lap times compared."""
    from src import simulate
    t = simulate._data()["teams"].set_index(["constructor_id", "season"])
    rel = ["constructor_last5_avg_finish_pct", "constructor_last5_avg_grid_pct", "constructor_points_share_before_race",
           "constructor_win_rate_last20", "constructor_podium_rate_last20", "constructor_dnf_rate_last10"]
    for c in pool["cars"]:
        v = t.loc[(c["constructor_id"], c["season"]), rel].astype(float)
        # grid percentile can exceed 1 slightly: a car starting behind the classified field (pit-lane start)
        cap = [1.1 if "grid" in k else 1.0 for k in rel]
        assert ((v >= 0) & (v <= cap)).all(), (c["constructor_id"], c["season"])
    a, b = t.loc[("mclaren", 1988), rel].astype(float), t.loc[("mercedes", 2020), rel].astype(float)
    assert a["constructor_points_share_before_race"] > 0.5 and b["constructor_points_share_before_race"] > 0.4   # both dominant in their era
    skills = [d["net_race_skill"] for d in pool["drivers"]]
    assert all(math.isfinite(x) for x in skills) and max(skills) - min(skills) < 10


@pytest.fixture(scope="module")
def app_grid(pool):
    """The app's race (D-036): 22 drivers, 11 constructors x 2, each constructor in its peak car."""
    return [{"driver_id": e["driver_id"], "constructor_id": e["constructor_id"]} for e in pool["default_grid"]]


def _swap_drivers(g, a, b):
    out = [dict(e) for e in g]
    i, j = [k for k, e in enumerate(out) if e["driver_id"] in (a, b)]
    out[i]["driver_id"], out[j]["driver_id"] = out[j]["driver_id"], out[i]["driver_id"]
    return out


def test_cross_era_grid_predicts_with_both_selected_models(client, app_grid):
    """Senna (form 1988) in the peak Mercedes and Hamilton (form 2020) in the peak McLaren, with three model pairs."""
    peak = {c["constructor_id"]: c for c in client.get("/api/driver-pool").json()["teams"]}
    e = _swap_drivers(app_grid, "senna", "hamilton")
    for cls, reg in (("random_forest", "ridge"), ("svm", "gradient_boosting_regression"), ("knn", "lasso")):
        r = _post(client, e, cls=cls, reg=reg)
        assert r.status_code == 200, r.text
        o = r.json()
        st = {s["driver_id"]: s for s in o["standings"]}
        assert o["models"]["classification"]["key"] == cls and o["models"]["regression"]["key"] == reg
        assert st["senna"]["constructor_id"] == "mercedes" and st["senna"]["constructor_season"] == peak["mercedes"]["season"]
        assert st["hamilton"]["constructor_name"] == "McLaren" and st["hamilton"]["season"] == 2020
        assert sorted(s["position"] for s in o["standings"]) == list(range(1, 23)) and len(set(o["podium"])) == 3
        assert all(0 <= s["podium_prob"] <= 1 and math.isfinite(s["expected_position"]) for s in o["standings"])
    # the car is a real input: the same Senna swapped into the weakest car on this grid drops
    weakest = min({x["constructor_id"] for x in e}, key=lambda c: peak[c]["peak_score"])
    worse = _swap_drivers(e, "senna", next(x["driver_id"] for x in e if x["constructor_id"] == weakest))
    a = next(s for s in _post(client, e).json()["standings"] if s["driver_id"] == "senna")
    b = next(s for s in _post(client, worse).json()["standings"] if s["driver_id"] == "senna")
    assert b["expected_position"] > a["expected_position"] and b["podium_prob"] < a["podium_prob"]


@pytest.mark.parametrize("entry, why", [
    ({"driver_id": "fangio"}, "not in the prediction pool"),
    ({"driver_id": "doohan"}, "not in the prediction pool"),
    ({"driver_id": "senna", "season": 1983}, "form season"),
    ({"driver_id": "alesi", "constructor_id": "ferrari", "constructor_season": 1975}, "peak"),
    ({"driver_id": "alesi", "constructor_id": "nowhere", "constructor_season": 2020}, "eligible constructors"),
])
def test_unsupported_combinations_are_rejected_with_a_reason(client, app_grid, entry, why):
    e = [dict(x) for x in app_grid]
    k = next((i for i, x in enumerate(e) if x["driver_id"] == entry["driver_id"]), 0)   # in place if already on the grid
    e[k] = {"constructor_id": e[k]["constructor_id"], **entry}
    r = _post(client, e)
    assert r.status_code == 422 and why in r.json()["detail"]["message"]


def test_a_constructor_takes_exactly_two_drivers_and_only_its_peak_car(client, app_grid, pool):
    three = [dict(x) for x in app_grid]
    three[2]["constructor_id"] = three[0]["constructor_id"]                  # 3 in one team, 1 in another
    name = next(c["constructor_name"] for c in pool["teams"] if c["constructor_id"] == three[0]["constructor_id"])
    r = _post(client, three)
    assert r.status_code == 422 and f"{name}: 3" in r.json()["detail"]["message"]
    peak = next(c["season"] for c in pool["teams"] if c["constructor_id"] == app_grid[0]["constructor_id"])
    other = [dict(x) for x in app_grid]
    other[0]["constructor_season"] = peak - 1                               # another season of the same team: not offered
    assert _post(client, other).status_code == 422
    assert _post(client, app_grid).status_code == 200
