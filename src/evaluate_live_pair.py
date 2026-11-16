"""Chronological replay of the default APEX pair on observed grids.

Run python -m src.evaluate_live_pair. The seasons were examined in prior work.
Refits are temporary and keep installed model settings and schemas fixed.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from src.algorithms import REGISTRY
from src.config import PROJECT_ROOT
from src.driver_ratings import attach, crew_per_race, load_ratings
from src.features import load_features
from src.multi_sim import _algos, bundle_score, ensemble_key, ensemble_values, position_summary
from src.predict import Predictor
from src.ranking import _positions, scenario_matrix
from src.train_algorithms import fit_tau, TARGET
from src.train_scenario import metrics, v1_system

PAIRS = {"default_pair": ("random_forest", "ridge"), "demo_pair": ("logistic_regression", "ridge")}
SEED, DRAWS = 42, 5000
CALIBRATION = (2017, 2018, 2019)
EVALUATION = (2023, 2024, 2025)
GRID = "grid_baseline"


def fit_bundle(df, key, settings, first, end):
    target = TARGET[settings["family"]]
    train = df[df.season.ge(first) & df.season.lt(end) & df[target].notna()]
    model = REGISTRY[key]["build"](**settings["params"])
    if "model__n_jobs" in model.get_params():
        model.set_params(model__n_jobs=1)
    model.fit(scenario_matrix(train, settings["features"]), train[target])
    return settings | {"model": model}


def ordered_grid(rows):
    rows = rows.sort_values(["race_id", "grid_position", "driver_id"], na_position="last").copy()
    rows["grid_position"] = rows.groupby("race_id").cumcount() + 1
    return rows


def calibrated(df, key, settings):
    first = settings["train_start"]
    cal = []
    for season in CALIBRATION:
        rows = ordered_grid(df[(df.season == season) & ~df.exclude_from_winner_metrics])
        rows["score"] = -np.log(rows.grid_position) if key == GRID else bundle_score(
            fit_bundle(df, key, settings, first, season), rows
        )
        cal.append(rows)
    cal = pd.concat(cal)
    return fit_tau(cal.score.to_numpy(), cal.race_id.to_numpy(), cal.target_winner.to_numpy())


def evaluate(df, bundles):
    test = ordered_grid(df[df.season.isin(EVALUATION)])
    scores = {k: bundle_score(b, test) * b["tau"] for k, b in bundles.items()}
    scores[GRID] = -np.log(test.grid_position.to_numpy()) * calibrated(df, GRID, {"train_start": 1981})
    outputs = {k: [] for k in (*scores, *PAIRS)}
    for indices in test.groupby("race_id", sort=False).indices.values():
        race = test.iloc[indices]
        noise = np.random.default_rng(SEED).gumbel(size=(DRAWS, len(race)))
        views = {k: np.array(position_summary(_positions(u[indices][None, :] + noise))) for k, u in scores.items()}
        views.update({name: ensemble_values(*(views[k] for k in pair)) for name, pair in PAIRS.items()})
        for k, (exp, win, pod) in views.items():
            assert np.isclose(win.sum(), 1) and np.isclose(pod.sum(), min(3, len(race)))
            assert np.all(win <= pod)
            ids = race.driver_id.to_numpy()
            if k in PAIRS:
                order = sorted(range(len(race)), key=lambda i: ensemble_key(exp[i], pod[i], ids[i]))
            else:
                order = sorted(range(len(race)), key=lambda i: (exp[i], -win[i], ids[i]))
            rank = np.empty(len(race), dtype=int)
            rank[order] = np.arange(1, len(race) + 1)
            outputs[k].append(race.assign(win_prob=win, podium_prob=pod, pred_rank=rank))
    return pd.DataFrame([{"model": k, **metrics(pd.concat(rows))} for k, rows in outputs.items()])


def main():
    tables = PROJECT_ROOT / "outputs" / "tables"
    raw = load_features()
    df = crew_per_race(attach(raw, load_ratings(), -1))
    active, _ = _algos()
    bundles = {}
    manifest = {}
    for key in dict.fromkeys(key for pair in PAIRS.values() for key in pair):
        installed = active[key]
        tau = calibrated(df, key, installed)
        bundles[key] = fit_bundle(df, key, installed, installed["train_start"], 2023) | {"tau": tau}
        artifact = PROJECT_ROOT / "models" / "algorithms" / f"{key}.joblib"
        manifest[key] = {k: installed[k] for k in ("params", "features", "feature_set", "train_start")}
        manifest[key] |= {"source_artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "refit_tau": round(tau, 12)}
    result = evaluate(df, bundles)
    result.insert(1, "protocol", "fixed fit through 2022; previously examined 2023–2025")
    result.to_csv(tables / "live_pair_replay.csv", index=False, float_format="%.12g")
    (tables / "live_pair_replay.json").write_text(json.dumps({
        "evaluated": EVALUATION, "training_end": 2022, "calibration": CALIBRATION,
        "selection": "installed default settings, fixed before this run; prior project work used these seasons",
        "grid": "observed ordering, unknown grid positions last, driver ID breaks ties",
        "inputs": "race-specific pre-start features with historical reanalysis; full observed fields",
        "baseline": "negative log grid position calibrated on 2017–2019",
        "seed": SEED, "draws": DRAWS, "models": manifest, "pairs": PAIRS,
    }, indent=2), encoding="utf-8")
    p = Predictor()
    corrected = v1_system(raw[raw.season.isin(EVALUATION)], p._models(2025), p.meta["tau"], p.meta["podium_method"], coherent=True)
    assert (corrected.win_prob <= corrected.podium_prob).all()
    pd.DataFrame([{"model": "legacy_coherent_replay", **metrics(corrected)}]).to_csv(
        tables / "historical_coherent_replay.csv", index=False, float_format="%.12g"
    )
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
