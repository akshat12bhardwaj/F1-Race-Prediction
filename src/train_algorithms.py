"""
Train and fairly evaluate every course algorithm on the race task (D-022).

Protocol (same for every algorithm; declared before results):
- Data: car-neutral scenario features (algorithms.FEATURES), seasons >= 1981 (SVM: >= 1995, O(n^2) training cost).
  Classification target: podium. Regression target: finishing percentile. Preprocessing inside each pipeline.
- Walk-forward: for S in 2017..2022 the model is trained on seasons < S and scores season S (out-of-sample).
- Calibration seasons 2017-2019: choose the hyper-parameters from a small grid AND fit the race temperature tau
  (maximum likelihood of the actual winners under softmax(tau * score)).
- Comparison seasons 2020-2022 (65 races): every metric reported, with config and tau fixed beforehand.
- The existing Plackett-Luce simulator model and a grid-only baseline go through the same protocol.
- "Best validated" = lowest winner log-loss on 2020-2022 (shown as an option, never forced; slightly optimistic
  because it is chosen on those seasons).
- Production: chosen config refit on all seasons up to 2025; tau refit on all 2017-2022 walk-forward scores.

- D-033: --features v1 (no weather beyond the wet flag) or v2 (+ pre-start weather); --out writes artefacts to another
  folder. A chronological TEST on 2023-2025 (train < S, config and tau fixed beforehand) is reported once, never
  used to choose anything.

Run:  python -m src.train_algorithms [--features v2 --out models/algorithms_wx/v2]     (~12 minutes)
"""

import json
import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

from src.algorithms import BASELINE_INFO, FEATURE_SETS, RANKING_INFO, REGISTRY, SEED, score
from src.config import PROJECT_ROOT, load_config
from src.driver_ratings import attach, crew_per_race, load_ratings
from src.evaluate import race_table, summarize_races
from src.features import load_features
from src.ranking import ScenarioRanker, race_probabilities, scenario_matrix

WF = [2017, 2018, 2019, 2020, 2021, 2022]
CALIB, EVAL, TEST = [2017, 2018, 2019], [2020, 2021, 2022], [2023, 2024, 2025]
START = 1981
TARGET = {"classification": "target_podium", "regression": "finish_pct"}


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def winner_nll(tau, s, race_ids, winner):
    d = pd.DataFrame({"r": race_ids, "u": tau * s, "w": winner})
    m = d.groupby("r")["u"].transform("max")
    lse = np.log(np.exp(d["u"] - m).groupby(d["r"]).transform("sum")) + m
    return -(d["u"] - lse)[d["w"] == 1].mean()


def fit_tau(s, race_ids, winner):
    r = minimize_scalar(lambda lt: winner_nll(np.exp(lt), s, race_ids, winner), bounds=(-6, 6), method="bounded")
    return float(np.exp(r.x))


def race_metrics(df, u, n_sims=2000):
    """Race-level metrics from strengths u (Plackett-Luce with the REAL grid already inside the features)."""
    o = df[["race_id", "season", "driver_id", "grid_position", "target_finish_position", "target_winner",
            "target_podium", "exclude_from_winner_metrics"]].copy()
    o["u"] = u
    win, pod = np.empty(len(o)), np.empty(len(o))
    for i, (_, idx) in enumerate(o.groupby("race_id").indices.items()):
        p = race_probabilities(o["u"].to_numpy()[idx], n_sims, SEED + i)
        win[idx], pod[idx] = p["win"], p["podium"]
    o["win_prob"], o["podium_prob"] = win, pod
    o["pred_rank"] = o.assign(n=-o["u"]).sort_values(["race_id", "n", "grid_position", "driver_id"]) \
        .groupby("race_id").cumcount().add(1).reindex(o.index)
    s = summarize_races(race_table(o))
    return {**{k: s[k] for k in ("winner_logloss", "winner_top1", "winner_top3", "podium_overlap", "exact_podium",
                                 "mean_spearman", "rank_mae")},
            "win_brier": brier_score_loss(o["target_winner"], o["win_prob"]),
            "podium_brier": brier_score_loss(o["target_podium"], o["podium_prob"]),
            "podium_ece": ece(o["target_podium"], o["podium_prob"])}


def ece(y, p, bins=10):
    """Expected calibration error: |mean predicted - observed| averaged over 10 equal-width probability bins."""
    b = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "b": b}).groupby("b")
    return float((d["p"].mean() - d["y"].mean()).abs().mul(d.size() / len(b)).sum())


def native_metrics(family, model, X, df):
    if family == "classification":
        s = score(model, family, X)
        hit = (s > 0).astype(int)        # score is a logit / log-ratio / margin: > 0 means p(podium) > 0.5
        out = {"podium_roc_auc": roc_auc_score(df["target_podium"], s),
               "podium_pr_auc": average_precision_score(df["target_podium"], s),
               "podium_precision": precision_score(df["target_podium"], hit, zero_division=0),
               "podium_recall": recall_score(df["target_podium"], hit, zero_division=0),
               "podium_f1": f1_score(df["target_podium"], hit, zero_division=0)}
        if hasattr(model, "predict_proba") and not type(model.steps[-1][1]).__name__ == "SVC":
            out["podium_brier_native"] = brier_score_loss(df["target_podium"], model.predict_proba(X)[:, 1])
        return out
    pred = model.predict(X)
    return {"finish_pct_mae": mean_absolute_error(df["finish_pct"], pred), "finish_pct_r2": r2_score(df["finish_pct"], pred),
            "finish_pct_rmse": float(np.sqrt(mean_squared_error(df["finish_pct"], pred))),
            "finish_pct_spearman": spearmanr(df["finish_pct"], pred).statistic}


def main(only=None, fset="v1", out=None):
    """only: list of algorithm keys to (re)train; the other results are kept from the previous run.
    fset: feature set (algorithms.FEATURE_SETS). out: artefact folder (default models/algorithms; tables go there too)."""
    cfg = load_config()
    FEATURES = FEATURE_SETS[fset]
    out_dir = PROJECT_ROOT / (out or (cfg["paths"]["models"] + "/algorithms"))
    tables = out_dir if out else PROJECT_ROOT / cfg["paths"]["tables"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = crew_per_race(attach(load_features(), load_ratings(), -1))
    df = df[df["season"] >= START].reset_index(drop=True)
    X_all = scenario_matrix(df, FEATURES)
    rows, perm_rows, meta = [], [], {}

    def evaluate(key, info, fit_fn, start):
        """fit_fn(train_df, X_train) -> (model, scorer(X, df)); returns record."""
        best = None
        for params in info["configs"]:
            scores, models = pd.Series(np.nan, index=df.index), {}
            t0 = time.perf_counter()
            for S in WF:
                tr = (df["season"] >= start) & (df["season"] < S) & df[info["target"]].notna()
                te = df["season"] == S
                m, scorer = fit_fn(params, df[tr], X_all[tr])
                scores[te] = scorer(X_all[te], df[te])
                models[S] = (m, scorer)
            fit_s = (time.perf_counter() - t0) / len(WF)
            cal = df["season"].isin(CALIB)
            keep = cal & ~df["exclude_from_winner_metrics"]
            tau = fit_tau(scores[keep].to_numpy(), df.loc[keep, "race_id"].to_numpy(), df.loc[keep, "target_winner"].to_numpy())
            cal_ll = winner_nll(tau, scores[keep].to_numpy(), df.loc[keep, "race_id"].to_numpy(),
                                df.loc[keep, "target_winner"].to_numpy())
            log(f"  {key:30s} {json.dumps(params):40s} tau={tau:7.3f} calib winner_ll={cal_ll:.4f} ({fit_s:.1f}s/fit)")
            if best is None or cal_ll < best["cal_ll"]:
                best = {"params": params, "tau": tau, "cal_ll": cal_ll, "scores": scores, "models": models, "fit_s": fit_s}
        ev = df["season"].isin(EVAL)
        rec = {"algorithm": key, "label": info["label"], "family": info["family"], "topic": info["topic"],
               "params": json.dumps(best["params"]), "tau": best["tau"], "calibration_winner_logloss": best["cal_ll"],
               "train_start": start, "mean_fit_seconds": round(best["fit_s"], 2),
               **race_metrics(df[ev], best["tau"] * best["scores"][ev].to_numpy())}
        return rec, best

    # ---------------- course algorithms ----------------
    for key, spec in REGISTRY.items():
        if only and key not in only:
            continue
        fam = spec["family"]
        target = TARGET[fam]
        configs = [dict(zip(spec["grid"], v)) for v in product(*spec["grid"].values())] or [{}]
        start = spec.get("train_start", START)

        def fit_fn(params, tr_df, X_tr, spec=spec, fam=fam, target=target):
            m = spec["build"](**params).fit(X_tr, tr_df[target])
            return m, (lambda X, d, m=m: score(m, fam, X))

        log(f"{key} ({fam})")
        rec, best = evaluate(key, {**spec, "configs": configs, "target": target}, fit_fn, start)
        ev = df["season"].isin(EVAL)
        m2020 = best["models"][2020][0]                     # trained on seasons < 2020: out-of-sample for 2020-2022
        rec.update(native_metrics(fam, m2020, X_all[ev], df[ev]))
        scoring = "roc_auc" if fam == "classification" else "neg_mean_absolute_error"
        y_ev = df.loc[ev, "target_podium" if fam == "classification" else "finish_pct"]
        pi = permutation_importance(m2020, X_all[ev], y_ev, scoring=scoring, n_repeats=3, random_state=SEED, n_jobs=4)
        for f, mu, sd in zip(FEATURES, pi.importances_mean, pi.importances_std):
            perm_rows.append({"algorithm": key, "feature": f, "importance_mean": mu, "importance_std": sd})
        # chronological test 2023-2025: config and tau fixed above; reported once, never used for selection
        ts = pd.Series(np.nan, index=df.index)
        for S in TEST:
            trS = (df["season"] >= start) & (df["season"] < S) & df[target].notna()
            mS = spec["build"](**best["params"]).fit(X_all[trS], df.loc[trS, target])
            ts[df["season"] == S] = score(mS, fam, X_all[df["season"] == S])
            if S == TEST[0]:
                m2023 = mS
        te = df["season"].isin(TEST)
        rec.update({f"test_{k}": v for k, v in race_metrics(df[te], best["tau"] * ts[te].to_numpy()).items()})
        rec.update({f"test_{k}": v for k, v in native_metrics(fam, m2023, X_all[te], df[te]).items()})
        rows.append(rec)
        # production fit + explanation artefacts
        tr = (df["season"] >= start) & df[target].notna()
        prod = spec["build"](**best["params"]).fit(X_all[tr], df.loc[tr, target])
        wf = df["season"].isin(WF) & ~df["exclude_from_winner_metrics"]
        tau_prod = fit_tau(best["scores"][wf].to_numpy(), df.loc[wf, "race_id"].to_numpy(), df.loc[wf, "target_winner"].to_numpy())
        bundle = {"model": prod, "family": fam, "tau": tau_prod, "params": best["params"], "features": FEATURES,
                  "explain": spec["explain"], "train_start": start, "feature_set": fset, "seed": SEED,
                  "data_cutoff": int(df["season"].max()), "trained": time.strftime("%Y-%m-%d %H:%M:%S")}
        if spec["explain"] == "neighbours":
            bundle["train_meta"] = df.loc[tr, ["race_id", "season", "race_name", "driver_id", "driver_first_name",
                                              "driver_last_name", "constructor_name", "grid_position",
                                              "target_finish_position", "target_podium"]].reset_index(drop=True)
        joblib.dump(bundle, out_dir / f"{key}.joblib", compress=3)
        meta[key] = {k: v for k, v in rec.items()} | {"tau_production": tau_prod, "params": best["params"]}

    # ---------------- Plackett-Luce default and grid baseline (same protocol) ----------------
    for key, kind, fs, info in () if fset != "v1" else (("plackett_luce", "pl_linear", "carneutral", RANKING_INFO),
                                ("grid_baseline", "pl_grid", None, BASELINE_INFO)):
        if only and key not in only:
            continue
        def fit_fn(params, tr_df, X_tr, kind=kind, fs=fs):
            m = ScenarioRanker(kind, half_life_seasons=8, top_k=1, feature_set=fs).fit(tr_df)
            return m, (lambda X, d, m=m: m.utility(d))
        log(key)
        rec, best = evaluate(key, {**info, "configs": [{}], "target": "target_finish_position"}, fit_fn, START)
        rows.append(rec)
        prod = ScenarioRanker(kind, half_life_seasons=8, top_k=1, feature_set=fs).fit(df)
        wf = df["season"].isin(WF) & ~df["exclude_from_winner_metrics"]
        tau_prod = fit_tau(best["scores"][wf].to_numpy(), df.loc[wf, "race_id"].to_numpy(), df.loc[wf, "target_winner"].to_numpy())
        joblib.dump({"model": prod, "family": info["family"], "tau": tau_prod, "params": {}, "features": FEATURES,
                     "explain": info["explain"], "train_start": START}, out_dir / f"{key}.joblib", compress=3)
        meta[key] = {k: v for k, v in rec.items()} | {"tau_production": tau_prod, "params": {}}

    comp = pd.DataFrame(rows)
    perm = pd.DataFrame(perm_rows)
    if only:   # merge with the previous full run
        old = pd.read_csv(tables / "algorithm_comparison.csv")
        comp = pd.concat([old[~old["algorithm"].isin(only)].drop(columns="best_validated"), comp], ignore_index=True)
        oldp = pd.read_csv(tables / "algorithm_permutation_importance.csv")
        perm = pd.concat([oldp[~oldp["algorithm"].isin(only)], perm], ignore_index=True)
        meta = json.loads((out_dir / "metadata.json").read_text())["algorithms"] | meta
    best_key = comp.sort_values("winner_logloss").iloc[0]["algorithm"]
    comp["best_validated"] = comp["algorithm"] == best_key
    comp.to_csv(tables / "algorithm_comparison.csv", index=False)
    perm.to_csv(tables / "algorithm_permutation_importance.csv", index=False)
    (out_dir / "metadata.json").write_text(json.dumps(
        {"best_validated": best_key, "protocol": __doc__, "calibration_seasons": CALIB, "evaluation_seasons": EVAL,
         "test_seasons": TEST, "features": FEATURES, "feature_set": fset, "seed": SEED, "algorithms": meta, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2, default=lambda x: None if x != x else float(x)))
    log(f"best validated: {best_key}")
    log("\n" + comp[["algorithm", "family", "tau", "winner_logloss", "winner_top1", "podium_overlap", "mean_spearman",
                     "rank_mae"]].round(4).sort_values("winner_logloss").to_string(index=False))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated algorithm keys to retrain")
    ap.add_argument("--features", default="v1", choices=list(FEATURE_SETS))
    ap.add_argument("--out", help="artefact folder relative to the project (default: models/algorithms)")
    a = ap.parse_args()
    main(a.only.split(",") if a.only else None, a.features, a.out)
