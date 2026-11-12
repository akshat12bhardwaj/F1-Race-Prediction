"""
Phase 7: local REST API (FastAPI) over the prediction engine.

Run:  python -m uvicorn src.api:app --port 8010
If frontend/dist exists (npm run build), the website is also served at http://127.0.0.1:8010/.

Outcomes are served ONLY by /api/race-results (reveal after prediction); /api/predict never
reads them.
"""

import json
import math
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, load_config
from src import multi_sim, simulate
from src.predict import ModelsMissing, Predictor, RaceUnavailable, race_summary

CFG = load_config()
TABLES = PROJECT_ROOT / CFG["paths"]["tables"]
FIGURES = PROJECT_ROOT / CFG["paths"]["figures"]

app = FastAPI(title="F1 Race Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def predictor():
    return Predictor()


def _p():
    try:
        return predictor()
    except (ModelsMissing, FileNotFoundError) as e:
        raise HTTPException(503, f"Models or data not available: {e}. Run python -m src.train") from e


def clean(obj):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def _check_race(p, season, rnd):
    status = p.season_status(season)
    if status == "locked_test":
        raise HTTPException(403, {"code": status, "message":
                                  f"{season} is a reserved test season, locked until the approved final evaluation."})
    if not any(r["round"] == rnd for r in p.races(season)):
        raise HTTPException(404, f"No model-eligible race for season {season} round {rnd}.")


@app.get("/api/health")
def health():
    try:
        p = predictor()
        return {"status": "ok", "model_version": p.meta["version"], "selected_models": p.meta["selected_models"],
                "replay_seasons": sorted(p.replay_seasons), "test_evaluated": p.meta.get("test_evaluated", False)}
    except (ModelsMissing, FileNotFoundError) as e:
        return {"status": "degraded", "detail": str(e)}


@app.get("/api/seasons")
def seasons():
    return _p().seasons()


@app.get("/api/races")
def races(season: int = Query(..., ge=1950, le=2100)):
    p = _p()
    r = p.races(season)
    if not r:
        raise HTTPException(404, f"No races for season {season}.")
    return clean({"season": season, "status": p.season_status(season), "races": r})


@app.get("/api/drivers")
def drivers(season: int = Query(..., ge=1950, le=2100), round: int = Query(..., ge=1, le=30)):
    p = _p()
    try:
        return clean({"season": season, "round": round, "drivers": p.drivers(season, round)})
    except RaceUnavailable as e:
        raise HTTPException(404, str(e)) from e


class PredictRequest(BaseModel):
    season: int = Field(..., ge=1950, le=2100)
    round: int = Field(..., ge=1, le=30)
    driver_ids: list[str] | None = Field(None, description="drivers to highlight; the full field is always modelled")


@app.post("/api/predict")
def predict(req: PredictRequest):
    p = _p()
    _check_race(p, req.season, req.round)
    field = {d["driver_id"] for d in p.drivers(req.season, req.round)}
    sel = req.driver_ids if req.driver_ids else sorted(field)
    unknown = sorted(set(sel) - field)
    if unknown:
        raise HTTPException(422, {"code": "unknown_drivers", "message":
                                  f"Not in this race's starting field: {', '.join(unknown)}"})
    pred = p.predict_race(req.season, req.round)
    pred["selected"] = pred["driver_id"].isin(sel)
    cols = ["driver_id", "driver_code", "driver_first_name", "driver_last_name", "driver_number",
            "constructor_id", "constructor_name", "grid_position", "qualifying_position", "win_prob",
            "podium_prob", "pred_position", "pred_rank", "selected", "factors", "key_inputs"]
    r = pred.iloc[0]
    return clean({
        "race": {"season": int(r["season"]), "round": int(r["round"]), "race_id": r["race_id"],
                 "race_name": r["race_name"], "circuit_name": r["circuit_name"], "country": r["country"],
                 "race_date": pd.Timestamp(r["race_date"]).strftime("%Y-%m-%d"), "starters": len(pred)},
        "model": {**p.model_info(req.season),
                  "selected_models": p.meta["selected_models"], "tau": p.meta["tau"],
                  "podium_method": p.meta["podium_method"]},
        "predictions": pred[cols].to_dict("records"),
        "podium": pred.nsmallest(3, "pred_rank")["driver_id"].tolist(),
        "checks": {"win_prob_sum": float(pred["win_prob"].sum()), "podium_prob_sum": float(pred["podium_prob"].sum())},
    })


@app.get("/api/race-results")
def race_results(season: int = Query(..., ge=1950, le=2100), round: int = Query(..., ge=1, le=30)):
    """Actual outcome for reveal-after-prediction. Same availability rules as /api/predict."""
    p = _p()
    _check_race(p, season, round)
    actual = p.actual_results(season, round)
    pred = p.predict_race(season, round, explain=False)
    return clean({"results": actual.to_dict("records"), "summary": race_summary(pred, actual)})


@app.get("/api/models")
def models():
    return clean(_p().meta)


def _csv(name):
    path = TABLES / name
    if not path.exists():
        raise HTTPException(503, f"{name} not found; run python -m src.train / python -m src.analysis")
    return pd.read_csv(path).replace({np.nan: None}).to_dict("records")


@app.get("/api/model-comparison")
def model_comparison():
    out = {"comparison": _csv("model_comparison.csv"), "paired_bootstrap": _csv("model_paired_bootstrap.csv"),
           "podium_method": _csv("e4_podium_method.csv")}
    for extra in ("feature_importance.csv", "calibration_curve.csv", "ablation_groups.csv",
                  "e11_numpy_vs_sklearn.csv", "season_metrics.csv", "scenario_model_comparison.csv",
                  "simulator_race_comparison.csv", "simulator_quali_comparison.csv", "simulator_post_hoc_2023_2025.csv",
                  "algorithm_comparison.csv", "algorithm_permutation_importance.csv", "car_clusters.csv",
                  "car_cluster_summary.csv",
                  "scenario_paired_bootstrap.csv", "scenario_subset_experiment.csv", "final_test_2023_2025.csv",
                  "final_test_by_season.csv"):
        if (TABLES / extra).exists():
            out[extra.removesuffix(".csv")] = _csv(extra)
    return clean(out)


# ---------------- Mode B: custom race simulator ----------------
def _sim(fn, *a, **k):
    try:
        return fn(*a, **k)
    except simulate.ScenarioError as e:
        raise HTTPException(422, {"code": "invalid_scenario", "message": str(e)}) from e
    except FileNotFoundError as e:
        raise HTTPException(503, f"Simulator model or data missing: {e}") from e


@app.get("/api/circuits")
def circuits():
    return clean(_sim(simulate.circuits))


@app.get("/api/available-drivers")
def available_drivers():
    return clean(_sim(simulate.available_drivers))


@app.get("/api/driver-performance-periods")
def driver_performance_periods(driver_id: str = Query(..., min_length=1, max_length=64)):
    return clean({"driver_id": driver_id, "periods": _sim(simulate.driver_periods, driver_id)})


@app.get("/api/constructors")
def constructors(season: int | None = Query(None, ge=1950, le=2100)):
    return clean(_sim(simulate.constructors, season))


@app.get("/api/default-grid")
def default_grid(circuit_id: str = Query(..., min_length=1, max_length=64)):
    """The eligible 2025 field in 2025 cars (reference season; see /api/driver-pool)."""
    return clean(_sim(simulate.default_grid, circuit_id))


@app.get("/api/driver-pool")
def driver_pool():
    """Every eligible driver (each in their form season) and every eligible car-season, 1981-2025 (D-032), plus the app's
    race (D-036): `teams` = one peak car per constructor, `default_grid` = the default 22-car grid, `grid` = the rule."""
    return clean(_sim(lambda: simulate.history_pool() | {
        "teams": simulate.peak_cars(), "default_grid": simulate.peak_default_grid(),
        "grid": {"drivers": simulate.GRID_SIZE, "constructors": simulate.GRID_TEAMS, "per_constructor": simulate.SEATS,
                 "peak_weights": simulate.PEAK_WEIGHTS, "peak_min_races": simulate.PEAK_MIN_RACES}}))


COND = {"weather", "rain_mm_h", "air_temp_c", "humidity_pct", "wind_kmh"}


class Entry(BaseModel):
    driver_id: str = Field(..., min_length=1, max_length=64)
    season: int | None = Field(None, ge=1950, le=2100, description="driver performance period (default: prime)")
    constructor_id: str | None = Field(None, max_length=64, description="car (default: the car of that season)")
    constructor_season: int | None = Field(None, ge=1950, le=2100)


class Scenario(BaseModel):
    circuit_id: str = Field(..., min_length=1, max_length=64)
    entries: list[Entry] = Field(..., min_length=2, max_length=simulate.MAX_DRIVERS)
    weather: str = Field("dry", pattern="^(dry|damp|wet)$", description="preset; explicit values below override it")
    rain_mm_h: float | None = Field(None, ge=0, le=4, description="rain in the hours before the start (mm/h)")
    air_temp_c: float | None = Field(None, ge=10, le=33, description="air temperature (not track temperature), °C")
    humidity_pct: float | None = Field(None, ge=20, le=95)
    wind_kmh: float | None = Field(None, ge=0, le=35, description="wind speed, km/h")
    n_sims: int = Field(5000, ge=100, le=simulate.MAX_SIMS)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    algorithm: str = Field("plackett_luce", max_length=40, description="algorithm key, or 'best' (best validated)")
    grid_mode: str = Field("qualifying", pattern="^(qualifying|fixed)$",
                           description="'fixed': the entry order is the starting grid; 'qualifying': simulate qualifying")


@app.post("/api/simulate-custom-race")
def simulate_custom_race(s: Scenario):
    return clean(_sim(simulate.predict_custom_race, s.circuit_id, [e.model_dump() for e in s.entries],
                      s.model_dump(include=COND), s.n_sims, s.seed, s.algorithm, s.grid_mode))


@app.get("/api/conditions")
def conditions():
    """Race-condition controls: ranges seen in 1981-2025 races, presets, and which inputs the installed models use."""
    used = sorted({w for m in _sim(multi_sim.catalog) for w in m["weather_inputs"]})
    return {"ranges": simulate.COND_RANGES, "presets": simulate.COND_PRESETS, "wet_mm_h": simulate.WET_MM_H,
            "min_humidity_rain": simulate.MIN_HUMIDITY_RAIN, "controls": [k for k in simulate.COND_CONTROLS if k in used or k == "rain_mm_h"]}


@app.get("/api/algorithms")
def algorithms():
    """Course algorithms available to the simulator, with explanations and real evaluation metrics."""
    return clean(_sim(multi_sim.catalog))


class Comparison(Scenario):
    algorithms: list[str] | None = Field(None, max_length=30, description="algorithm keys; omit or ['all'] for every one")


@app.post("/api/compare-algorithms")
def compare_algorithms(c: Comparison):
    """The same custom race predicted by several algorithms with shared random numbers."""
    if c.algorithms is not None and len(c.algorithms) == 0:
        raise HTTPException(422, {"code": "invalid_scenario", "message": "Select at least one algorithm."})
    return clean(_sim(multi_sim.simulate_algorithms, c.circuit_id, [e.model_dump() for e in c.entries],
                      c.algorithms, c.model_dump(include=COND), c.n_sims, c.seed, c.grid_mode))


class Explain(Scenario):
    driver_id: str = Field(..., min_length=1, max_length=64)


@app.post("/api/explain-algorithm")
def explain_algorithm(x: Explain):
    """Model-specific explanation of one driver's prediction by one algorithm."""
    return clean(_sim(multi_sim.explain_driver, x.circuit_id, [e.model_dump() for e in x.entries], x.algorithm,
                      x.driver_id, x.model_dump(include=COND), x.grid_mode))


class RacePrediction(BaseModel):
    circuit_id: str = Field(..., min_length=1, max_length=64)
    entries: list[Entry] = Field(..., min_length=3, max_length=simulate.MAX_DRIVERS,
                                 description="in starting-grid order (first = pole)")
    weather: str = Field("dry", pattern="^(dry|damp|wet)$", description="preset; explicit values below override it")
    rain_mm_h: float | None = Field(None, ge=0, le=4, description="rain in the hours before the start (mm/h)")
    air_temp_c: float | None = Field(None, ge=10, le=33, description="air temperature (not track temperature), °C")
    humidity_pct: float | None = Field(None, ge=20, le=95)
    wind_kmh: float | None = Field(None, ge=0, le=35, description="wind speed, km/h")
    classification: str = Field(..., max_length=40)
    regression: str = Field(..., max_length=40)
    n_sims: int = Field(5000, ge=100, le=simulate.MAX_SIMS)
    seed: int = Field(42, ge=0, le=2**31 - 1)


@app.post("/api/race-prediction")
def race_prediction(r: RacePrediction):
    """The app's prediction: one classification + one regression model, podium / not on podium / full standings."""
    t0 = time.perf_counter()
    entries = _sim(simulate.check_app_grid, [e.model_dump() for e in r.entries])   # 22 drivers, 11 peak cars x 2
    out = _sim(multi_sim.predict_pair, r.circuit_id, entries, r.classification,
               r.regression, r.model_dump(include=COND), r.n_sims, r.seed)
    # measured server compute time for this request (models are loaded once and reused; nothing is retrained)
    return clean(out | {"inference_ms": round(1000 * (time.perf_counter() - t0), 1)})


@app.get("/api/simulator-model")
def simulator_model():
    meta = _sim(lambda: simulate._data()["meta"])
    out = {k: v for k, v in meta.items() if k != "coefficients_standardised"} | {"coefficients": meta["coefficients_standardised"]}
    v1 = PROJECT_ROOT / CFG["paths"]["models"] / "scenario" / "metadata.json"   # earlier scenario model (D-020)
    if v1.exists():
        out["v1_scenario_selected"] = json.loads(v1.read_text())["selected"]["name"]
    return clean(out)


DIST = PROJECT_ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
