"""Custom race simulator v2: full default grid, prime form, car swaps, estimated grid (engine + API).
Needs: python -m src.snapshots, python -m src.driver_ratings, python -m src.train_simulator."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT
from src.features import load_features

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/scenario/simulator_v2.joblib").exists(),
                                reason="simulator v2 not trained")


@pytest.fixture(scope="module")
def sim():
    from src import simulate
    return simulate


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def monaco():
    from conftest import prime_field
    return {"entries": prime_field("monaco"), "grid_capacity": 26}


def _post(client, entries, circuit="monaco", **kw):
    return client.post("/api/simulate-custom-race", json={"circuit_id": circuit, "entries": entries, **kw})


def _by_id(res):
    return {r["driver_id"]: r for r in res["results"]}


def _swap(entries, a, b):
    e = [dict(x) for x in entries]
    ia = next(i for i, x in enumerate(e) if x["driver_id"] == a)
    ib = next(i for i, x in enumerate(e) if x["driver_id"] == b)
    for k in ("constructor_id", "constructor_season"):
        e[ia][k], e[ib][k] = e[ib][k], e[ia][k]
    return e


# ---------------- defaults: full grid, prime form, default cars ----------------
def test_default_grid_is_full_capacity_with_prime_form(client, monaco, sim):
    caps = {c["circuit_id"]: c["grid_capacity"] for c in client.get("/api/circuits").json()}
    assert caps["monaco"] == 26 and caps["miami"] == 20 and caps["baku"] == 22 and max(caps.values()) <= 26
    assert monaco["grid_capacity"] == 26 and len(monaco["entries"]) == 26
    primes = pd.read_csv(PROJECT_ROOT / "data/processed/driver_primes.csv").set_index("driver_id")
    for e in monaco["entries"]:
        p = primes.loc[e["driver_id"]]
        assert e["season"] == p["prime_season"] and e["constructor_id"] == p["prime_constructor_id"]
        assert e["constructor_season"] == e["season"] and p["greats_eligible"]
    # the app's default grid is the eligible 2025 field at every circuit
    miami = client.get("/api/default-grid", params={"circuit_id": "miami"}).json()
    assert len(miami["entries"]) == 20 and "max_verstappen" in {e["driver_id"] for e in miami["entries"]}


def test_prime_is_best_rated_season_with_enough_starts():
    snaps = pd.read_csv(PROJECT_ROOT / "data/processed/driver_snapshots.csv")
    from src.driver_ratings import attach, load_ratings
    s = attach(snaps, load_ratings(), 0)
    s = s[(s["driver_id"] == "alonso") & (s["season_starts"] >= 8)]
    score = s["net_quali_skill"] + s["net_race_skill"]
    primes = pd.read_csv(PROJECT_ROOT / "data/processed/driver_primes.csv").set_index("driver_id")
    assert primes.loc["alonso", "prime_season"] == s.loc[score.idxmax(), "season"]


def test_full_default_grid_simulation_is_coherent(client, monaco):
    r = _post(client, monaco["entries"])
    assert r.status_code == 200, r.text
    b = r.json()
    res = b["results"]
    assert len(res) == 26 and sorted(x["pred_rank"] for x in res) == list(range(1, 27))
    assert sorted(x["est_grid"] for x in res) == list(range(1, 27))            # estimated grid is a permutation
    assert sum(x["win_prob"] for x in res) == pytest.approx(1, abs=1e-9)
    assert sum(x["podium_prob"] for x in res) == pytest.approx(3, abs=1e-9)
    assert sum(x["top5_prob"] for x in res) == pytest.approx(5, abs=1e-9)
    assert sum(x["pole_prob"] for x in res) == pytest.approx(1, abs=1e-9)
    exp = [x["expected_position"] for x in res]
    assert exp == sorted(exp)                                                   # order = expected finish
    assert b["podium"] == [x["driver_id"] for x in res[:3]]
    for x in res:
        assert sum(x["position_distribution"]) == pytest.approx(1) and x["factors"]
    assert any("Cross-era" in w for w in b["warnings"])


# ---------------- car swaps ----------------
def test_swapping_cars_swaps_exactly_the_car_inputs(sim, monaco):
    e = monaco["entries"]
    a, _, _ = sim.build_scenario("monaco", e)
    b, _, _ = sim.build_scenario("monaco", _swap(e, "senna", "hunt"))
    a, b = a.set_index("driver_id"), b.set_index("driver_id")
    car = [c for c in sim.CAR_COLS] + ["constructor_id", "constructor_season", "constructor_circuit_avg_finish_pct"]
    drv = sim.DRIVER_COLS + ["season", "driver_circuit_starts"]
    for c in car:
        assert a.loc["senna", c] == b.loc["hunt", c] or (pd.isna(a.loc["senna", c]) and pd.isna(b.loc["hunt", c])), c
        assert a.loc["hunt", c] == b.loc["senna", c] or (pd.isna(a.loc["hunt", c]) and pd.isna(b.loc["senna", c])), c
    for c in drv:                                                               # driver inputs unchanged
        for d in ("senna", "hunt"):
            assert a.loc[d, c] == b.loc[d, c] or (pd.isna(a.loc[d, c]) and pd.isna(b.loc[d, c])), c
    others = [d for d in a.index if d not in ("senna", "hunt")]
    pd.testing.assert_frame_equal(a.loc[others, car + drv], b.loc[others, car + drv])


def test_car_swap_changes_predictions_coherently_and_reproducibly(client, monaco):
    base = _by_id(_post(client, monaco["entries"]).json())
    swapped_entries = _swap(monaco["entries"], "senna", "hunt")       # McLaren 1988 <-> Hesketh 1974
    s1, s2 = _post(client, swapped_entries).json(), _post(client, swapped_entries).json()
    assert s1["results"] == s2["results"] and s1["scenario_id"] == s2["scenario_id"]
    s = _by_id(s1)
    assert s["senna"]["constructor_name"] == "Hesketh" and s["hunt"]["constructor_name"] == "McLaren"
    assert s["senna"]["win_prob"] < base["senna"]["win_prob"]                    # weaker car -> weaker result
    assert s["hunt"]["win_prob"] > base["hunt"]["win_prob"]
    assert s["senna"]["est_grid"] > base["senna"]["est_grid"] and s["hunt"]["est_grid"] < base["hunt"]["est_grid"]
    assert sum(x["win_prob"] for x in s1["results"]) == pytest.approx(1)
    assert sorted(x["pred_rank"] for x in s1["results"]) == list(range(1, 27))
    assert s1["scenario_id"] != _post(client, monaco["entries"]).json()["scenario_id"]


def test_any_car_season_can_be_assigned_and_is_flagged(client, monaco):
    e = [dict(x) for x in monaco["entries"]]
    e[0].update(constructor_id="minardi", constructor_season=1990)
    b = _post(client, e).json()
    r = _by_id(b)[e[0]["driver_id"]]
    assert r["constructor_name"] == "Minardi" and r["constructor_season"] == 1990 and not r["real_pairing"]
    assert any("never raced together" in w for w in b["warnings"])


# ---------------- sensitivity ----------------
def test_circuit_weather_and_snapshot_sensitivity(client, monaco):
    e = monaco["entries"]
    w = lambda res: {r["driver_id"]: r["win_prob"] for r in res["results"]}
    base = w(_post(client, e).json())
    assert base != w(_post(client, e, circuit="monza").json())
    assert base != w(_post(client, e, weather="wet").json())
    e2 = [dict(x) for x in e]
    i = next(k for k, x in enumerate(e2) if x["driver_id"] == "alonso")
    e2[i]["season"] = 2023                                                     # Alonso 2023 form, same car
    assert w(_post(client, e2).json())["alonso"] != base["alonso"]


def test_seed_controls_monte_carlo(client, monaco):
    a = _post(client, monaco["entries"], seed=1).json()
    b = _post(client, monaco["entries"], seed=2).json()
    assert a["results"] != b["results"]
    assert max(abs(x["win_prob"] - _by_id(b)[x["driver_id"]]["win_prob"]) for x in a["results"]) < 0.05


def test_small_custom_field_never_exceeds_participants(client):
    b = _post(client, [{"driver_id": "hamilton"}, {"driver_id": "alonso"}]).json()
    assert sum(x["podium_prob"] for x in b["results"]) == pytest.approx(2) and len(b["podium"]) == 2
    assert all(x["season"] == pd.read_csv(PROJECT_ROOT / "data/processed/driver_primes.csv")
               .set_index("driver_id").loc[x["driver_id"], "prime_season"] for x in b["results"])


# ---------------- catalogues & lookups ----------------
def test_catalogues(client):
    d = {x["driver_id"]: x for x in client.get("/api/available-drivers").json()}
    assert d["hamilton"]["prime_season"] in d["hamilton"]["seasons"] and len(d) > 600
    p = client.get("/api/driver-performance-periods", params={"driver_id": "alonso"}).json()["periods"]
    assert sum(x["is_prime"] for x in p) == 1 and 2002 not in {x["season"] for x in p}
    t = client.get("/api/constructors", params={"season": 1988}).json()
    assert t[0]["constructor_id"] == "mclaren"


def test_snapshot_matches_training_feature_code():
    snaps = pd.read_csv(PROJECT_ROOT / "data/processed/driver_snapshots.csv")
    mf = load_features()
    first_2021 = mf[(mf["race_id"] == "2021_01") & (mf["driver_id"] == "hamilton")].iloc[0]
    s = snaps[(snaps["driver_id"] == "hamilton") & (snaps["season"] == 2020)].iloc[0]
    for c in ["driver_last5_avg_finish_pct", "driver_teammate_finish_gap_last10", "driver_teammate_grid_gap_last10",
              "driver_teammate_beat_rate", "driver_career_starts"]:
        assert s[c] == pytest.approx(first_2021[c]), c


def test_circuit_history_matches_training_features(sim):
    rows, _, _ = sim.build_scenario("monaco", [{"driver_id": "hamilton", "season": 2020},
                                               {"driver_id": "leclerc", "season": 2020}])
    mf = load_features()
    real = mf[(mf["race_id"] == "2021_05") & (mf["driver_id"] == "hamilton")].iloc[0]
    assert rows.set_index("driver_id").loc["hamilton", "driver_circuit_starts"] == real["driver_circuit_starts"]


# ---------------- validation ----------------
@pytest.mark.parametrize("entries,circuit,msg", [
    ([{"driver_id": "hamilton"}, {"driver_id": "nobody"}], "monaco", "Unknown driver"),
    ([{"driver_id": "hamilton"}, {"driver_id": "hamilton", "season": 2019}], "monaco", "Duplicate driver"),
    ([{"driver_id": "hamilton"}, {"driver_id": "alonso", "season": 2002}], "monaco", "did not start"),
    ([{"driver_id": "hamilton"}, {"driver_id": "alonso", "constructor_id": "mclaren", "constructor_season": 1950}],
     "monaco", "No car data"),
    ([{"driver_id": "hamilton"}, {"driver_id": "alonso"}], "atlantis", "Unknown circuit"),
])
def test_invalid_scenarios(client, entries, circuit, msg):
    r = _post(client, entries, circuit=circuit)
    assert r.status_code == 422 and msg in r.json()["detail"]["message"]


def test_capacity_and_schema_validation(client):
    from conftest import prime_field
    full = prime_field("monaco")
    # Miami only ever held 20 cars in the data; the app's fixed 22-car format raises every circuit to at least 22 (D-036)
    assert _post(client, full[:22], circuit="miami").status_code == 200
    miami = _post(client, full[:23], circuit="miami")
    assert miami.status_code == 422 and "at most 22" in miami.json()["detail"]["message"]
    assert _post(client, full[:1]).status_code == 422
    assert _post(client, full[:4], weather="snow").status_code == 422
    assert _post(client, full[:4], n_sims=10).status_code == 422
    assert client.get("/api/default-grid", params={"circuit_id": "atlantis"}).status_code == 422


def test_historical_replay_still_works_and_labels_mode(client):
    r = client.post("/api/predict", json={"season": 2024, "round": 1}).json()
    assert r["model"]["mode"] == "out_of_sample" and r["model"]["trained_on"].endswith("2022")
    r = client.post("/api/predict", json={"season": 2010, "round": 1}).json()
    assert r["model"]["mode"] == "retrospective"
