"""Prediction engine + REST API tests (needs trained models: python -m src.train)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT
from src.features import TARGETS, load_features

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/metadata_v1.json").exists(),
                                reason="models not trained")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


def _predict(client, season=2022, rnd=1, ids=None):
    return client.post("/api/predict", json={"season": season, "round": rnd, "driver_ids": ids})


def test_health_and_seasons(client):
    h = client.get("/api/health").json()
    # 2023-2025 unlocked after the one-time test evaluation (D-020); <= 2016 = labelled retrospective fits
    assert h["status"] == "ok" and h["replay_seasons"] == [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    st = {s["season"]: s["status"] for s in client.get("/api/seasons").json()}
    assert st[2022] == st[2017] == st[2023] == st[2025] == "available"
    assert st[2010] == "retrospective" and st[1950] == "retrospective"


def test_races_and_drivers(client):
    r = client.get("/api/races", params={"season": 2022}).json()
    assert len(r["races"]) == 22 and r["status"] == "available"
    d = client.get("/api/drivers", params={"season": 2022, "round": 1}).json()["drivers"]
    assert len(d) == 20 and d[0]["grid_position"] == 1
    assert all("target_finish_position" not in x and "position_text" not in x for x in d)


@pytest.mark.parametrize("season,rnd", [(2017, 1), (2019, 10), (2020, 5), (2022, 1), (2022, 22)])
def test_predict_is_coherent(client, season, rnd):
    r = _predict(client, season, rnd)
    assert r.status_code == 200, r.text
    body = r.json()
    p = pd.DataFrame(body["predictions"])
    n = len(p)
    assert sorted(p["pred_rank"]) == list(range(1, n + 1))
    assert p["win_prob"].sum() == pytest.approx(1, abs=1e-9)
    assert p["podium_prob"].sum() == pytest.approx(3, abs=1e-9) and p["podium_prob"].max() <= 1 + 1e-9
    assert body["podium"] == p.sort_values("pred_rank")["driver_id"].head(3).tolist()
    assert not any(k.startswith("target_") or k in ("official_position", "points", "status") for k in p.columns)
    assert body["model"]["trained_on"].endswith(str(season - 1))


def test_subset_does_not_change_field_probabilities(client):
    full = pd.DataFrame(_predict(client).json()["predictions"]).set_index("driver_id")
    sub = _predict(client, ids=["leclerc", "hamilton"]).json()
    s = pd.DataFrame(sub["predictions"]).set_index("driver_id")
    assert set(s.index[s["selected"]]) == {"leclerc", "hamilton"} and len(s) == len(full)
    np.testing.assert_allclose(s.loc[full.index, "win_prob"], full["win_prob"])


def test_repeat_predictions_identical(client):
    a, b = _predict(client).json(), _predict(client).json()
    assert a["predictions"] == b["predictions"]


def test_errors(client):
    assert _predict(client, ids=["not_a_driver"]).status_code == 422
    assert _predict(client, ids=["verstappen_typo", "leclerc"]).json()["detail"]["code"] == "unknown_drivers"
    assert _predict(client, 2010, 1).json()["model"]["mode"] == "retrospective"
    assert _predict(client, 2022, 29).status_code == 404
    assert client.post("/api/predict", json={"season": "x"}).status_code == 422
    assert client.post("/api/predict", json={"season": 2022, "round": 0}).status_code == 422
    assert client.get("/api/races", params={"season": 1900}).status_code == 422
    assert client.get("/api/drivers", params={"season": 2022, "round": 25}).status_code == 404


def test_race_results_reveal(client):
    r = client.get("/api/race-results", params={"season": 2022, "round": 1}).json()
    assert r["results"][0]["driver_id"] == "leclerc"          # 2022 Bahrain GP winner
    assert set(r["summary"]["actual_podium"]) == {"leclerc", "sainz", "hamilton"}
    assert 0 <= r["summary"]["podium_overlap"] <= 3


def test_models_and_comparison(client):
    m = client.get("/api/models").json()
    assert m["test_evaluated"] is True and set(m["selected_models"]) == {"winner", "podium", "position"}
    c = client.get("/api/model-comparison").json()
    models = {(r["task"], r["model"]) for r in c["comparison"]}
    for task in ("winner", "podium"):
        assert {(task, x) for x in ("logistic_regression", "decision_tree", "random_forest",
                                    "gradient_boosting", "grid_heuristic", "dummy_prior")} <= models
    assert {("position", x) for x in ("linear_regression", "ridge", "lasso", "decision_tree",
                                      "random_forest", "gradient_boosting", "grid_heuristic")} <= models


def test_inference_never_reads_outcomes():
    """Scrambling every outcome column must leave predictions unchanged."""
    from src.predict import Predictor
    mf = load_features()
    scrambled = mf.copy()
    rng = np.random.default_rng(0)
    for c in TARGETS:
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    a = Predictor().predict_race(2021, 22)
    b = Predictor(features=scrambled).predict_race(2021, 22)
    pd.testing.assert_frame_equal(a.drop(columns=["factors", "key_inputs"]), b.drop(columns=["factors", "key_inputs"]))


def test_test_season_lock_logic(client):
    """Before the test evaluation, 2023-2025 must be refused by predict and race-results."""
    from src.api import predictor
    p = predictor()
    saved = set(p.replay_seasons)
    try:
        p.replay_seasons -= p.test_seasons
        assert _predict(client, 2023, 1).status_code == 403
        assert _predict(client, 2023, 1).json()["detail"]["code"] == "locked_test"
        assert client.get("/api/race-results", params={"season": 2024, "round": 1}).status_code == 403
    finally:
        p.replay_seasons = saved
