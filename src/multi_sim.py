"""
Multi-algorithm race simulation (D-022): the same custom race predicted by every course algorithm.

Common random numbers: all algorithms share the sampled qualifying grids AND the race noise, so differences
between their predictions come from the algorithms only.

Speed-up (exact): a driver's features depend on the simulated grid only through the grid slot, so each algorithm
scores an N x N table (every driver at every slot) once; Monte Carlo races look scores up in that table.
"""

import hashlib
import json
import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.algorithms import BASELINE_INFO, FEATURES as ALG_FEATURES, GROUPS_V1 as GROUPS, RANKING_INFO, REGISTRY, score as alg_score   # weather is race-level: no per-driver group effect
from src.config import PROJECT_ROOT, load_config
from src.ranking import GROUP_LABELS, _positions, scenario_matrix
from src.simulate import ScenarioError, _data, build_scenario, capacity

log = logging.getLogger("apex.prediction")

ALGO_ORDER = ["plackett_luce", "logistic_regression", "decision_tree", "knn", "svm", "naive_bayes", "random_forest",
              "bagging", "adaboost", "gradient_boosting", "neural_network", "linear_regression", "ridge", "lasso",
              "decision_tree_regression", "random_forest_regression", "gradient_boosting_regression", "grid_baseline"]
INPUT_COLS = ["net_quali_skill", "net_race_skill", "driver_teammate_beat_rate", "driver_career_starts",
              "constructor_last5_avg_grid_pct", "constructor_last5_avg_finish_pct", "constructor_points_share_before_race",
              "constructor_dnf_rate_last10", "driver_circuit_starts"]
RAW_OF = {"log_grid": "grid_position", "front_row": "grid_position", "pole": "grid_position",
          "log_grid_x_pole_rate": "grid_position", "log_grid_x_wet": "grid_position",
          "log_driver_career_starts": "driver_career_starts", "log_driver_circuit_starts": "driver_circuit_starts"}


@lru_cache(maxsize=1)
def _algos():
    d = PROJECT_ROOT / load_config()["paths"]["models"] / "algorithms"
    if not (d / "metadata.json").exists():
        raise FileNotFoundError("models/algorithms missing; run python -m src.train_algorithms")
    meta = json.loads((d / "metadata.json").read_text())
    bundles = {k: joblib.load(d / f"{k}.joblib") for k in ALGO_ORDER if (d / f"{k}.joblib").exists()}
    for b in bundles.values():   # deterministic predictions (parallel tree averaging can jitter by 1 ulp)
        est = getattr(b["model"], "steps", [[None, None]])[-1][1]
        if est is not None and hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)
    return bundles, meta


# model inputs behind the user's condition controls: v2 bundles read all four (D-033), v1 only the wet flag
WEATHER_INPUTS = ["rain_mm_h", "air_temp_c", "humidity_pct", "wind_kmh"]


def info(key):
    return REGISTRY.get(key) or (RANKING_INFO if key == "plackett_luce" else BASELINE_INFO)


@lru_cache(maxsize=None)
def _artifact_sha(key):
    path = PROJECT_ROOT / load_config()["paths"]["models"] / "algorithms" / f"{key}.joblib"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog():
    bundles, meta = _algos()
    out = []
    for k in ALGO_ORDER:
        if k not in bundles:
            continue
        i, m = info(k), meta["algorithms"].get(k, {})
        out.append({"key": k, "label": i["label"], "family": i["family"], "topic": i["topic"], "how": i["how"],
                    "strengths": i["strengths"], "weaknesses": i["weaknesses"], "sensitivity": i["sensitivity"],
                    "explain": i["explain"], "params": m.get("params", {}), "tau": bundles[k]["tau"],
                    "artifact_sha256": _artifact_sha(k), "feature_schema": bundles[k].get("feature_set", i["family"]),
                    "training_end": bundles[k].get("data_cutoff", 2025),
                    "evaluation": {"start": 2020, "end": 2022, "races": 61,
                                   "race_metrics": "annual refits before each evaluation season",
                                   "native_metrics": "single fit before 2020",
                                   "provenance": "active model configuration; chronological evaluation refits"},
                    "train_start": bundles[k]["train_start"], "best_validated": k == meta["best_validated"],
                    "weather_inputs": WEATHER_INPUTS if bundles[k].get("feature_set") == "v2" else [] if k == "grid_baseline" else ["wet"],
                    "metrics": {x: m.get(x) for x in (
                        "winner_logloss", "winner_top1", "winner_top3", "podium_overlap", "exact_podium",
                        "mean_spearman", "rank_mae", "podium_brier", "podium_roc_auc", "podium_pr_auc",
                        "finish_pct_mae", "finish_pct_r2", "calibration_winner_logloss", "mean_fit_seconds")}})
    return out


def resolve(algorithms):
    bundles, meta = _algos()
    if algorithms in (None, [], "all", ["all"]):
        return [k for k in ALGO_ORDER if k in bundles]
    keys = [meta["best_validated"] if a == "best" else a for a in algorithms]
    bad = [k for k in keys if k not in bundles]
    if bad:
        raise ScenarioError(f"Unknown algorithm(s): {', '.join(bad)}. "
                            f"Available: {', '.join(k for k in ALGO_ORDER if k in bundles)}, best, all.")
    return list(dict.fromkeys(keys))


def bundle_score(b, rows):
    if b["family"] in ("ranking", "baseline"):
        return b["model"].utility(rows)
    return alg_score(b["model"], b["family"], scenario_matrix(rows, b["features"]))   # each bundle's own schema


def score_table(b, df):
    """tau * score of every driver at every grid slot 1..N."""
    n = len(df)
    rows = df.iloc[np.repeat(np.arange(n), n)].reset_index(drop=True)
    rows["grid_position"] = np.tile(np.arange(1, n + 1), n).astype(float)
    return b["tau"] * bundle_score(b, rows).reshape(n, n)


def grids(df, grid_mode, n_sims, rng):
    """(sampled grids n_sims x n, most likely grid). 'fixed': the entry order IS the grid in every simulation."""
    n = len(df)
    if grid_mode == "fixed":
        return np.tile(np.arange(1, n + 1), (n_sims, 1)), np.arange(1, n + 1)
    if grid_mode != "qualifying":
        raise ScenarioError("grid_mode must be 'qualifying' or 'fixed'.")
    u_q = _data()["models"]["quali"].utility(df)
    return (_positions(u_q[None, :] + rng.gumbel(size=(n_sims, n))),
            pd.Series(-u_q).rank(method="first").astype(int).to_numpy())


def outcomes(pos, win, df):
    """The distinct answers to 'who wins': most likely winner, most likely exact podium, one sampled race."""
    fin = np.argsort(pos, axis=1)                         # driver index at P1, P2, ... in each simulated race
    ids = df["driver_id"].to_numpy()
    k = min(3, pos.shape[1])
    combos, counts = np.unique(fin[:, :k], axis=0, return_counts=True)
    top = int(np.argmax(counts))
    w = int(np.argmax(win))
    return {"most_likely_winner": {"driver_id": ids[w], "win_prob": float(win[w])},
            "most_likely_podium": {"driver_ids": [ids[j] for j in combos[top]], "prob": float(counts[top] / len(pos))},
            "sample_race": [ids[j] for j in fin[0]]}


def position_summary(pos):
    return pos.mean(0), (pos == 1).mean(0), (pos <= min(3, pos.shape[1])).mean(0)


def ensemble_values(classification, regression):
    return (classification + regression) / 2


def ensemble_key(expected, podium, driver_id):
    return expected, -podium, driver_id


def simulate_algorithms(circuit_id, entries, algorithms=None, conditions=None, n_sims=5000, seed=42,
                        grid_mode="qualifying"):
    D = _data()
    bundles, meta = _algos()
    keys = resolve(algorithms)
    df, warnings, weather = build_scenario(circuit_id, entries, conditions, n_sims)
    n, n_sims = len(df), int(n_sims)
    rng = np.random.default_rng(int(seed))
    grid, est_grid = grids(df, grid_mode, n_sims, rng)                     # shared grids (qualifying stage)
    noise = rng.gumbel(size=(n_sims, n))                                    # shared race noise
    pole, exp_grid = (grid == 1).mean(axis=0), grid.mean(axis=0)
    est = dict(zip(df["driver_id"], est_grid))
    runs = {}
    for k in keys:
        table = score_table(bundles[k], df)
        pos = _positions(table[np.arange(n)[None, :], grid - 1] + noise)
        dist = np.stack([(pos == j).mean(axis=0) for j in range(1, n + 1)], axis=1)
        exp, win, podium = position_summary(pos)
        top5 = (pos <= min(5, n)).mean(0)
        order = sorted(range(n), key=lambda i: (exp[i], -win[i], df.loc[i, "driver_id"]))
        rank = np.empty(n, int)
        rank[order] = np.arange(1, n + 1)
        sens = group_sensitivity(bundles[k], df, est)
        res = []
        for i in order:
            r = df.loc[i]
            res.append({**{c: r[c] for c in ["driver_id", "driver_name", "driver_code", "driver_number", "season",
                                             "form_basis", "constructor_id", "constructor_name", "constructor_season",
                                             "car_basis", "eligibility", "evidence"]},
                        "factors": sens[r["driver_id"]],
                        "is_prime": bool(r["is_prime"]), "real_pairing": bool(r["real_pairing"]),
                        "est_grid": int(est_grid[i]), "expected_grid": float(exp_grid[i]), "pole_prob": float(pole[i]),
                        "pred_rank": int(rank[i]), "win_prob": float(win[i]), "podium_prob": float(podium[i]),
                        "top5_prob": float(top5[i]), "expected_position": float(exp[i]),
                        "prob_predicted_position": float(dist[i, rank[i] - 1]),
                        "win_prob_se": float(np.sqrt(win[i] * (1 - win[i]) / n_sims)),
                        "podium_prob_se": float(np.sqrt(podium[i] * (1 - podium[i]) / n_sims)),
                        "strength_at_est_grid": float(table[i, est_grid[i] - 1]),
                        "position_distribution": [float(x) for x in dist[i]],
                        "inputs": {c: (None if pd.isna(r[c]) else float(r[c])) for c in INPUT_COLS}})
        runs[k] = {"algorithm": k, "label": info(k)["label"], "family": info(k)["family"], "results": res,
                   "podium": [x["driver_id"] for x in res[:min(3, n)]],   # predicted order by expected position
                   **outcomes(pos, win, df),
                   "checks": {"win_prob_sum": float(win.sum()), "podium_prob_sum": float(podium.sum()),
                              "expected_podium_slots": min(3, n)}}
    circ = D["circuits"].set_index("circuit_id").loc[circuit_id]
    canon = json.dumps({"c": circuit_id, "e": sorted([[r["driver_id"], int(r["season"]), r["constructor_id"],
                                                       int(r["constructor_season"])] for _, r in df.iterrows()]),
                        "w": weather, "n": n_sims, "s": int(seed),
                        **({"g": list(df["driver_id"])} if grid_mode == "fixed" else {})}, sort_keys=True)
    return {"scenario_id": hashlib.sha1(canon.encode()).hexdigest()[:12],
            "circuit": {"circuit_id": circuit_id, "circuit_name": circ["circuit_name"], "country": circ["country"],
                        "race_name": circ["race_name"], "pole_win_rate": float(circ["circuit_pole_win_rate"]),
                        "races": int(circ["races"]), "historical_grid_capacity": int(circ["grid_capacity"]),
                        "grid_capacity": capacity(circuit_id, entries)},
            "conditions": weather, "n_sims": n_sims, "seed": int(seed), "n_drivers": n,
            "grid_mode": grid_mode,
            "qualifying": ("Fixed: your grid order is used in every simulated race." if grid_mode == "fixed" else
                           f"Simulated by the {D['meta']['quali_model']['name']} qualifying model for every race model; "
                           "the selected algorithm only predicts the race from that grid."),
            "warnings": warnings + density_warnings(df), "runs": runs, "algorithms": keys,
            "best_validated": meta["best_validated"], "agreement": agreement(runs, df) if len(runs) > 1 else None}


def agreement(runs, df):
    keys, ids = list(runs), list(df["driver_id"])
    rank = {k: pd.Series({r["driver_id"]: r["pred_rank"] for r in runs[k]["results"]}).reindex(ids) for k in keys}
    winp = {k: {r["driver_id"]: r["win_prob"] for r in runs[k]["results"]} for k in keys}
    votes = pd.Series([runs[k]["podium"][0] for k in keys]).value_counts()
    pvotes = pd.Series([d for k in keys for d in runs[k]["podium"]]).value_counts()
    M = np.array([[rank[a].corr(rank[b], method="spearman") for b in keys] for a in keys])
    per = []
    for d in ids:
        w = np.array([winp[k][d] for k in keys])
        p = np.array([rank[k][d] for k in keys])
        per.append({"driver_id": d, "mean_win_prob": float(w.mean()), "min_win_prob": float(w.min()),
                    "max_win_prob": float(w.max()), "best_rank": int(p.min()), "worst_rank": int(p.max()),
                    "mean_rank": float(p.mean())})
    per.sort(key=lambda x: x["mean_rank"])
    m = len(keys)
    return {"winner_votes": [{"driver_id": d, "votes": int(v)} for d, v in votes.items()],
            "podium_votes": [{"driver_id": d, "votes": int(v)} for d, v in pvotes.items()],
            "consensus_winner": votes.index[0], "consensus_share": float(votes.iloc[0] / m),
            "rank_correlation": {"algorithms": keys, "matrix": np.round(M, 4).tolist(),
                                 "mean_off_diagonal": float((np.nansum(M) - m) / max(1, m * m - m))},
            "per_driver": per}


def group_sensitivity(b, df, est):
    """Model-agnostic local sensitivity: change in this algorithm's race strength (tau * score, a log-odds scale
    shared by all algorithms) when one feature group of a driver is replaced by the field median. Not causal."""
    rows = df.assign(grid_position=df["driver_id"].map(est).astype(float))
    base = b["tau"] * bundle_score(b, rows)
    out = {d: [] for d in rows["driver_id"]}
    for g, cols in GROUPS.items():
        alt = rows.copy()
        for c in {RAW_OF.get(c, c) for c in cols}:
            if c in alt and alt[c].notna().any():
                med = alt[c].median()
                alt[c] = float(np.round(med)) if c == "grid_position" else med
        eff = base - b["tau"] * bundle_score(b, alt)
        eff = eff - np.mean(eff)
        for d, e in zip(rows["driver_id"], eff):
            out[d].append({"group": g, "label": GROUP_LABELS[g], "effect": round(float(e), 3)})
    for d in out:
        out[d].sort(key=lambda f: -abs(f["effect"]))
    return out


def predict_pair(circuit_id, entries, classification, regression, conditions=None, n_sims=5000, seed=42):
    """The product's prediction: ONE classification model + ONE regression model on the same race.

    Both race the identical scenario (the entry order is the starting grid; shared random numbers). Each yields a
    full simulated finishing distribution; the final order ranks drivers by the average of the two models'
    expected finishing positions (an equal-weight two-model ensemble), so podium and standings are one coherent
    answer. Each model's own view is returned alongside, plus whether their podiums agree."""
    for key, fam in ((classification, "classification"), (regression, "regression")):
        if REGISTRY.get(key, {}).get("family") != fam:
            raise ScenarioError(f"Choose a {fam} model ('{key}' is not one).")
    out = simulate_algorithms(circuit_id, entries, [classification, regression], conditions, n_sims, seed, "fixed")
    rc = {r["driver_id"]: r for r in out["runs"][classification]["results"]}
    rr = {r["driver_id"]: r for r in out["runs"][regression]["results"]}
    rows = []
    for d, c in rc.items():
        g = rr[d]
        rows.append({k: c[k] for k in ("driver_id", "driver_name", "driver_code", "driver_number", "season", "form_basis",
                                       "constructor_id", "constructor_name", "constructor_season", "car_basis",
                                       "real_pairing", "is_prime", "eligibility", "evidence")}
                    | {"grid": c["est_grid"],
                       "expected_position": ensemble_values(c["expected_position"], g["expected_position"]),
                       "podium_prob": ensemble_values(c["podium_prob"], g["podium_prob"]),
                       "win_prob": ensemble_values(c["win_prob"], g["win_prob"]),
                       "classification": {"podium_prob": c["podium_prob"], "expected_position": c["expected_position"],
                                          "rank": c["pred_rank"]},
                       "regression": {"podium_prob": g["podium_prob"], "expected_position": g["expected_position"],
                                      "rank": g["pred_rank"]},
                       "key_factor": c["factors"][0]["label"] if c["factors"] else None})
    rows.sort(key=lambda r: ensemble_key(r["expected_position"], r["podium_prob"], r["driver_id"]))
    for i, r in enumerate(rows, 1):
        r["position"] = i
    k = min(3, len(rows))
    pod_c, pod_r = out["runs"][classification]["podium"], out["runs"][regression]["podium"]
    for w in out["warnings"]:
        log.info("scenario %s: %s", out["scenario_id"], w)
    lvl = {lv: [r["driver_id"] for r in rows if r["eligibility"] == lv] for lv in ("limited", "insufficient")}
    return {"scenario_id": out["scenario_id"], "circuit": out["circuit"], "conditions": out["conditions"],
            "n_sims": out["n_sims"], "seed": out["seed"], "diagnostics": out["warnings"], "eligibility": lvl, "standings": rows,
            "podium": [r["driver_id"] for r in rows[:k]], "not_podium": [r["driver_id"] for r in rows[k:]],
            "models": {"classification": {"key": classification, "label": info(classification)["label"], "podium": pod_c},
                       "regression": {"key": regression, "label": info(regression)["label"], "podium": pod_r}},
            "podium_agreement": sorted(pod_c) == sorted(pod_r)}


def predict_single(circuit_id, entries, algorithm="plackett_luce", conditions=None, n_sims=5000, seed=42,
                   grid_mode="qualifying"):
    """One algorithm, backwards-compatible single-run response with per-driver factors."""
    if algorithm == "all":
        raise ScenarioError("Select one algorithm or use /api/compare-algorithms for all models.")
    out = simulate_algorithms(circuit_id, entries, [algorithm], conditions, n_sims, seed, grid_mode)
    key = out["algorithms"][0]
    run = out["runs"][key]
    meta, i = _data()["meta"], info(key)
    fixed = grid_mode == "fixed"
    return {**{k: v for k, v in out.items() if k not in ("runs", "agreement")}, **run,
            "methodology": {
                "model": ("your fixed grid" if fixed else f"{meta['quali_model']['name']} (qualifying)")
                         + f" → {i['label']} (race)",
                "trained_on": f"{_algos()[0][key]['train_start']}-2025",
                "summary": ("Every simulated race starts from your grid order. " if fixed else
                            "Each simulated race draws a qualifying order from the qualifying model. ")
                           + f"The finishing order comes from {i['label']}: its score for every driver at the grid slot is turned "
                           "into race probabilities with a race-level temperature fitted on validation seasons, and the "
                           "order is drawn Plackett–Luce style. Probabilities are frequencies over the simulated races.",
                "prime_rule": meta["prime_rule"], "grid_capacity_rule": meta["grid_capacity_rule"],
                "assumptions": [
                    "Driver skill is car-neutral: measured only against teammates in the same car (teammate-network "
                    "rating), so drivers with unusually strong teammates may be underrated.",
                    "Pairings that never raced have no data on driver–car chemistry; the algorithm combines driver "
                    "and car features as it learned from real pairings.",
                    "Car strength is relative to its own season's field, not an absolute lap time.",
                    "The race sampling step assumes the odds between two drivers do not depend on who else races.",
                ]}}


def explain_driver(circuit_id, entries, algorithm, driver_id, conditions=None, grid_mode="qualifying"):
    """Model-specific explanation of one driver's score in this scenario, at the estimated (or fixed) grid slot."""
    bundles, _ = _algos()
    if algorithm == "all":
        raise ScenarioError("Select one algorithm to explain.")
    key = resolve([algorithm])[0]
    b = bundles[key]
    df, _, _ = build_scenario(circuit_id, entries, conditions)
    if driver_id not in set(df["driver_id"]):
        raise ScenarioError(f"{driver_id} is not in this race.")
    est = dict(zip(df["driver_id"], grids(df, grid_mode, 1, np.random.default_rng(0))[1]))
    rows = df.assign(grid_position=df["driver_id"].map(est).astype(float))
    i = int(np.where(rows["driver_id"] == driver_id)[0][0])
    inf = info(key)
    out = {"algorithm": key, "label": inf["label"], "explain": inf["explain"], "driver_id": driver_id,
           "driver_name": rows.loc[i, "driver_name"], "est_grid": int(est[driver_id]),
           "local_sensitivity": group_sensitivity(b, df, est)[driver_id]}
    model = b["model"]
    if key in ("plackett_luce", "grid_baseline"):
        names = list(model.prep_.get_feature_names_out())
        z = model.prep_.transform(scenario_matrix(rows, model._cols()))[i]
        out["native"] = {"kind": "coefficients", "items": _top(names, model.pl_.coef_, b["tau"] * z * model.pl_.coef_, z),
                         "note": "contribution = temperature × weight × standardised value (log-strength)"}
        return out
    feats = b["features"]
    X = scenario_matrix(rows, feats)
    pre, est_ = model[:-1], model.steps[-1][1]
    Z = pre.transform(X) if len(model.steps) > 1 else X.to_numpy()
    names = list(pre.get_feature_names_out()) if len(model.steps) > 1 else feats
    kind = inf["explain"]
    if kind == "coefficients":
        coef = np.ravel(est_.coef_)
        sign = 1 if b["family"] == "classification" else -1               # regression score = -prediction
        out["native"] = {"kind": "coefficients", "items": _top(names, coef, b["tau"] * sign * Z[i] * coef, Z[i]),
                         "note": "contribution = temperature × weight × standardised value"
                                 + ("" if sign == 1 else "; sign flipped because a LOWER predicted finishing percentile is better")}
    elif kind == "tree_path":
        t = est_.tree_
        steps = []
        for nid in est_.decision_path(Z[i:i + 1]).indices:
            if t.children_left[nid] == t.children_right[nid]:
                val = t.value[nid][0]
                leaf = float(val[1] / val.sum()) if b["family"] == "classification" else float(val[0])
                steps.append({"leaf": True, "samples": int(t.n_node_samples[nid]), "value": leaf,
                              "meaning": "podium rate of the past driver-races in this leaf" if b["family"] == "classification"
                              else "average finishing percentile in this leaf (0 = winner, 1 = last)"})
            else:
                v, thr = float(Z[i, t.feature[nid]]), float(t.threshold[nid])
                steps.append({"leaf": False, "feature": names[t.feature[nid]], "threshold": thr, "value": v,
                              "goes": "≤" if v <= thr else ">"})
        out["native"] = {"kind": "tree_path", "steps": steps}
    elif kind == "neighbours":
        dist, idx = est_.kneighbors(Z[i:i + 1], n_neighbors=8)
        m = b["train_meta"].iloc[idx[0]]
        items = []
        for (_, r), dd in zip(m.iterrows(), dist[0]):
            items.append({c: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for c, v in r.items()}
                         | {"distance": float(dd)})
        out["native"] = {"kind": "neighbours", "k": int(est_.n_neighbors), "items": items,
                         "podium_share_of_8": float(m["target_podium"].mean())}
    elif kind == "class_means":
        th, var = est_.theta_, est_.var_
        gap = (th[1] - th[0]) / np.sqrt(var.mean(axis=0))
        order = np.argsort(-np.abs(gap))[:8]
        out["native"] = {"kind": "class_means", "items": [
            {"feature": names[j], "podium_mean": float(th[1, j]), "non_podium_mean": float(th[0, j]),
             "standardised_gap": float(gap[j]), "driver_value": float(Z[i, j])} for j in order]}
    else:
        out["native"] = {"kind": "global_importance"}
    perm_path = PROJECT_ROOT / load_config()["paths"]["tables"] / "algorithm_permutation_importance.csv"
    if perm_path.exists():   # measured on 2020-2022 with the model trained on seasons < 2020
        perm = pd.read_csv(perm_path)
        perm = perm[perm["algorithm"] == key].sort_values("importance_mean", ascending=False).head(8)
        out["permutation_importance"] = perm[["feature", "importance_mean", "importance_std"]].to_dict("records")
    imp = getattr(est_, "feature_importances_", None)
    if imp is None and hasattr(est_, "estimators_"):
        try:
            imp = np.mean([e.feature_importances_ for e in est_.estimators_], axis=0)
        except AttributeError:
            imp = None
    if imp is not None:
        order = np.argsort(-imp)[:10]
        out["impurity_importance"] = [{"feature": names[j], "importance": float(imp[j])} for j in order]
    return out


def _top(names, coef, contrib, z, k=10):
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": names[j], "weight": float(coef[j]), "standardised_value": float(z[j]),
             "contribution": float(contrib[j])} for j in order]


# ---------------- anomaly detection by density estimation ----------------
@lru_cache(maxsize=1)
def _density():
    p = PROJECT_ROOT / load_config()["paths"]["models"] / "algorithms" / "density.joblib"
    return joblib.load(p) if p.exists() else None


def density_warnings(df):
    d = _density()
    if d is None:
        return []
    # only pairings that never raced can be off the historical driver-car manifold; real prime pairings are rare
    # (extreme skill + strong car) but they are real data, so they are not flagged
    hyp = df[~df["real_pairing"]]
    if hyp.empty:
        return []
    ll = d["model"].score_samples(d["prep"].transform(scenario_matrix(hyp, d["features"])))
    odd = hyp.loc[ll < d["threshold"], "driver_name"].tolist()
    if not odd:
        return []
    return [f"Outside historical data: {len(odd)} hypothetical driver–car pairing(s) ({', '.join(odd[:6])}"
            f"{'…' if len(odd) > 6 else ''}) look less typical than {1 - d['quantile']:.0%} of real F1 entries "
            f"(Gaussian-mixture density estimate), so every algorithm is extrapolating for them."]
