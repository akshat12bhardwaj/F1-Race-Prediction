"""D-033: pre-start weather features, retrained models, condition controls. Real data and trained models; no mocks."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT, load_config

MODELS = PROJECT_ROOT / "models"
pytestmark = pytest.mark.skipif(not (MODELS / "algorithms_wx/v2/metadata.json").exists(), reason="weather models not trained")
WX = ["wx_prestart_precip_sum_mm", "wx_prestart_temp_mean_c", "wx_prestart_humidity_mean_pct", "wx_prestart_wind_speed_mean_kmh"]


@pytest.fixture(scope="module")
def races():
    from src.features import load_features
    mf = load_features()
    return mf[mf["season"] >= 1981].drop_duplicates("race_id")


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def grid(client):
    return client.get("/api/default-grid", params={"circuit_id": "spa"}).json()["entries"]


@pytest.fixture(scope="module")
def app_grid(client):
    """The app's race (D-036): the default 22-car grid, 11 constructors x 2 in their peak cars."""
    return client.get("/api/driver-pool").json()["default_grid"]


def _post(client, entries, **cond):
    return client.post("/api/race-prediction", json={"circuit_id": "spa", "entries": entries, "classification": cond.pop("cls", "random_forest"),
                                                     "regression": cond.pop("reg", "ridge"), "n_sims": 1500, **cond})


# ---------------- data ----------------
def test_prestart_weather_covers_every_era_with_plausible_units(races):
    assert len(races) >= 800
    for c in WX:
        assert races[c].notna().all(), c                                       # 100% coverage 1981-2025
    assert races.groupby(races["season"] // 10)[WX].apply(lambda d: d.notna().mean().min()).min() == 1.0
    assert races["wx_prestart_temp_mean_c"].between(-10, 45).all()             # °C, air at 2 m
    assert races["wx_prestart_humidity_mean_pct"].between(0, 100).all()        # %
    assert races["wx_prestart_precip_sum_mm"].between(0, 50).all()             # mm over the 2-hour window
    assert races["wx_prestart_wind_speed_mean_kmh"].between(0, 150).all()      # km/h at 10 m
    # the training wet flag is exactly the 0.5 mm / 2 h threshold the app's wet state uses
    thr = load_config()["weather"]["wet_precip_threshold_mm"]
    assert ((races["wx_prestart_precip_sum_mm"] >= thr) == (races["wx_prestart_wet"] == 1)).all()
    from src import simulate
    assert 2 * simulate.WET_MM_H == thr


def test_no_future_weather_or_track_temperature_is_used():
    from src.algorithms import FEATURE_SETS
    from src.ranking import WX_RAW
    cfg = load_config()["weather"]
    assert cfg["prestart_window_hours"][1] < 0                               # window ends before the start
    assert all(k.startswith("wx_prestart_") for k in WX_RAW)                    # never the race window (wx_race_*)
    assert not any(f.startswith(("wx_race", "race_")) or "track_temp" in f or "wind_dir" in f for f in FEATURE_SETS["v2"])
    assert FEATURE_SETS["v2"][:len(FEATURE_SETS["v1"])] == FEATURE_SETS["v1"]   # driver/car/circuit features unchanged


# ---------------- models ----------------
def test_both_feature_sets_trained_with_recorded_configuration():
    for fs in ("v1", "v2"):
        meta = json.loads((MODELS / f"algorithms_wx/{fs}/metadata.json").read_text())
        assert meta["feature_set"] == fs and meta["seed"] == 42 and meta["test_seasons"] == [2023, 2024, 2025]
        for k, m in meta["algorithms"].items():
            if m["family"] in ("classification", "regression"):
                for c in ("calibration_winner_logloss", "winner_logloss", "test_winner_logloss", "podium_brier", "podium_ece"):
                    assert np.isfinite(m[c]), (fs, k, c)
                if m["family"] == "classification":
                    assert all(0 <= m[c] <= 1 for c in ("podium_roc_auc", "podium_precision", "podium_recall", "podium_f1"))
                else:
                    assert m["finish_pct_rmse"] >= m["finish_pct_mae"] > 0
    import joblib
    b = joblib.load(MODELS / "algorithms_wx/v2/random_forest.joblib")
    assert b["feature_set"] == "v2" and b["data_cutoff"] == 2025 and b["seed"] == 42
    assert (MODELS / "algorithms_d023_backup/random_forest.joblib").exists()     # previous production kept


def test_pre_registered_rule_was_applied_to_every_model():
    comp = pd.read_csv(PROJECT_ROOT / "outputs/tables/weather_model_comparison.csv")
    meta = json.loads((MODELS / "algorithms/metadata.json").read_text())
    for k, g in comp.groupby("algorithm"):
        a, b = g.set_index("feature_set").loc["v1"], g.set_index("feature_set").loc["v2"]
        want = "v2" if b["calibration_winner_logloss"] < a["calibration_winner_logloss"] and b["winner_logloss"] <= a["winner_logloss"] + 0.01 else "v1"
        assert g.loc[g["chosen"], "feature_set"].item() == want == meta["chosen_feature_sets"][k], k


def test_installed_bundles_match_their_schema_and_inference_builds_it(grid):
    import joblib
    from src import simulate
    from src.algorithms import FEATURE_SETS
    from src.ranking import scenario_matrix
    df = simulate.build_scenario("spa", simulate.check_entries(grid), {"rain_mm_h": 1.2, "air_temp_c": 14, "humidity_pct": 88, "wind_kmh": 22})[0]
    assert (df["wx_prestart_precip_sum_mm"] == 2.4).all() and (df["wx_prestart_temp_mean_c"] == 14).all() and (df["wx_prestart_wet"] == 1).all()
    for f in (MODELS / "algorithms").glob("*.joblib"):
        b = joblib.load(f)
        if b.get("family") not in ("classification", "regression"):
            continue
        assert b["features"] == FEATURE_SETS[b["feature_set"]], f.name
        X = scenario_matrix(df, b["features"])
        assert list(X.columns) == b["features"] and X[[c for c in b["features"] if c in ("rain_log", "air_temp_c", "humidity_pct", "wind_kmh")]].notna().all().all()


def test_weather_changes_only_models_that_read_it(client, app_grid):
    grid = app_grid
    """No artificial effect: a temperature change moves a weather-aware model, and leaves a dry/wet-only model exactly unchanged."""
    cat = {m["key"]: m for m in client.get("/api/algorithms").json()}
    aware = [k for k, m in cat.items() if len(m["weather_inputs"]) > 1]
    blind = [k for k, m in cat.items() if m["weather_inputs"] == ["wet"] and m["family"] in ("classification", "regression")]
    for k in aware[:2] + blind[:2]:
        fam = cat[k]["family"]
        args = {"cls": k} if fam == "classification" else {"reg": k}
        cold = _post(client, grid, air_temp_c=10, **dict(args)).json()
        hot = _post(client, grid, air_temp_c=33, **dict(args)).json()
        mine = lambda o: [s[fam]["expected_position"] for s in o["standings"]]
        if k in aware:
            assert mine(cold) != mine(hot), k
        else:
            assert mine(cold) == mine(hot), k


def test_prediction_shape_and_reproducibility_under_conditions(client, app_grid):
    grid = app_grid
    a = _post(client, grid, weather="damp", air_temp_c=16, wind_kmh=30).json()
    b = _post(client, grid, weather="damp", air_temp_c=16, wind_kmh=30).json()
    strip = lambda o: {k: v for k, v in o.items() if k != "inference_ms"}
    assert strip(a) == strip(b)
    assert a["conditions"] == {"rain_mm_h": 0.1, "air_temp_c": 16.0, "humidity_pct": 75.0, "wind_kmh": 30.0, "weather": "damp"}
    assert len(set(a["podium"])) == 3 and sorted(s["position"] for s in a["standings"]) == list(range(1, 23))
    assert abs(sum(s["podium_prob"] for s in a["standings"]) - 3) < 1e-6


@pytest.mark.parametrize("cond, ok", [({"rain_mm_h": 4}, True), ({"rain_mm_h": 4.5}, False), ({"air_temp_c": 9}, False),
                                      ({"humidity_pct": 96}, False), ({"wind_kmh": -1}, False), ({"weather": "snow"}, False),
                                      ({"rain_mm_h": 1, "humidity_pct": 30}, False), ({"rain_mm_h": 0, "humidity_pct": 20}, True)])
def test_condition_ranges_and_impossible_combinations(client, app_grid, cond, ok):
    grid = app_grid
    assert (_post(client, grid, **cond).status_code == 200) == ok


def test_conditions_endpoint_offers_only_model_inputs(client):
    c = client.get("/api/conditions").json()
    cat = client.get("/api/algorithms").json()
    used = {w for m in cat for w in m["weather_inputs"]}
    assert set(c["controls"]) <= used | {"rain_mm_h"} and "rain_mm_h" in c["controls"]
    assert c["ranges"]["air_temp_c"] == [10.0, 33.0] and c["presets"]["dry"]["rain_mm_h"] == 0
