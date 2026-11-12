"""
Phases 3-5: tune, compare and select models with season-based chronological validation.

Protocol (D-017):
- Data: development seasons only (<= 2022). Test seasons 2023-2025 are never loaded here.
- Tuning: hyperparameters chosen on expanding-window season folds 2017..2021
  (train = all seasons before the fold season, from train_start).
- Walk-forward predictions: every model (tuned params) is refit for each season
  S in 2017..2022 on seasons < S and predicts S. These are out-of-sample.
- Selection (pre-declared): among ML models (not baselines), best pooled 2017-2022
  primary metric: winner/podium = PR-AUC, finishing position = MAE. 2022 alone reported too.
- Race-level coherence: win temperature tau fitted on 2017-2021 walk-forward predictions;
  podium method (E4) chosen by pooled podium overlap.
- Saved: walk-forward pipelines of the selected models (used for historical replay: a race in
  season S is always predicted by a model trained only on seasons < S) and final pipelines
  trained on <= 2022 (reserved for the approved one-time test evaluation).

Run:  python -m src.train          (about 15-30 minutes)
"""

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import PROJECT_ROOT, load_config
from src.evaluate import (best_f1_threshold, classification_metrics, final_order, fit_tau,
                          normalize_podium, normalize_win, paired_bootstrap_diff, race_table,
                          regression_metrics, summarize_races, bootstrap_ci)
from src.features import FEATURES, load_features
from src.models import SEED, classifier_specs, regressor_specs
from src.splits import development_rows, expanding_folds, walk_forward_seasons

TASKS = {"winner": ("target_winner", "clf"), "podium": ("target_podium", "clf"),
         "position": ("target_finish_position", "reg")}
PRIMARY = {"winner": ("pr_auc", True), "podium": ("pr_auc", True), "position": ("mae", False)}
KEEP = ["race_id", "season", "round", "driver_id", "constructor_id", "grid_position",
        "target_finish_position", "target_winner", "target_podium", "exclude_from_winner_metrics"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def tune(pipe, space, kind, X, y, folds, scoring):
    if not space:
        return {}, None
    n_jobs = 1 if "random_forest" in str(pipe) else -1
    if kind == "grid":
        s = GridSearchCV(pipe, space, cv=folds, scoring=scoring, n_jobs=n_jobs, refit=False)
    else:
        s = RandomizedSearchCV(pipe, space, n_iter=kind[1], cv=folds, scoring=scoring, n_jobs=n_jobs,
                               refit=False, random_state=SEED)
    s.fit(X, y)
    return s.best_params_, pd.DataFrame(s.cv_results_)


def predict(model, X, kind):
    return model.predict_proba(X)[:, 1] if kind == "clf" else model.predict(X)


def walk_forward(pipe, X, y, seasons, kind, start):
    """Refit for every season S on seasons < S; return out-of-sample predictions and models."""
    pred = pd.Series(np.nan, index=X.index)
    fitted, secs = {}, 0.0
    for S in seasons:
        tr = (X["season"] >= start) & (X["season"] < S)
        te = X["season"] == S
        t0 = time.perf_counter()
        m = clone(pipe).fit(X[tr], y[tr])
        secs += time.perf_counter() - t0
        pred[te] = predict(m, X[te], kind)
        fitted[S] = m
    return pred, fitted, secs / len(seasons)


def race_eval(df, task, col):
    """Race-level metrics for one task's prediction column on a frame of out-of-sample rows."""
    d = df[KEEP].copy()
    if task == "position":
        d["win_prob"] = 0.0   # unused: only rank metrics are kept for this task
        d["pred_rank"] = final_order(d.assign(score=df[col]), "score")
        rt = race_table(d)
        return {k: v for k, v in summarize_races(rt).items()
                if k in ("races", "podium_overlap", "exact_podium", "mean_spearman", "rank_mae")}, rt
    if task == "winner":
        d["win_prob"] = normalize_win(df[col], d["race_id"])
    else:
        d["win_prob"] = df[col]
    d["pred_rank"] = final_order(d.assign(score=-d["win_prob"]), "score")
    rt = race_table(d)
    keys = ("races", "winner_top1", "winner_top3", "winner_logloss") if task == "winner" \
        else ("races", "podium_overlap", "exact_podium")
    return {k: v for k, v in summarize_races(rt).items() if k in keys}, rt


def main():
    cfg = load_config()
    split = cfg["split"]
    start = split["train_start"]
    out_tables = PROJECT_ROOT / cfg["paths"]["tables"]
    out_pred = PROJECT_ROOT / cfg["paths"]["predictions"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    for p in (out_tables, out_pred, models_dir):
        p.mkdir(parents=True, exist_ok=True)

    df = development_rows(load_features())
    df = df[df["season"] >= start].reset_index(drop=True)
    assert df["season"].max() <= max(split["validation_years"]), "test seasons leaked into development"
    X = df[FEATURES + ["season"]]
    wf_seasons = walk_forward_seasons()
    cv_seasons = split["cv_validation_years"]
    folds = list(expanding_folds(df, cv_seasons, start))
    oos = df["season"].isin(wf_seasons)
    oof = df.loc[oos, KEEP].copy()
    rows, tuning, selected, race_tables, timings = [], [], {}, {}, {}
    fitted_all = {}

    for task, (target, kind) in TASKS.items():
        y = df[target]
        specs = classifier_specs(FEATURES) if kind == "clf" else regressor_specs(FEATURES)
        scoring = "average_precision" if kind == "clf" else "neg_mean_absolute_error"
        for name, (pipe, space, search, role) in specs.items():
            t0 = time.perf_counter()
            best, cvres = tune(pipe, space, search, X, y, folds, scoring)
            tune_s = time.perf_counter() - t0
            if cvres is not None:
                tuning.append(cvres.assign(task=task, model=name)[
                    ["task", "model", "params", "mean_test_score", "std_test_score", "rank_test_score"]])
            tuned = clone(pipe).set_params(**best)
            pred, fitted, fit_s = walk_forward(tuned, X, y, wf_seasons, kind, start)
            col = f"{task}__{name}"
            oof[col] = pred[oos].to_numpy()
            fitted_all[(task, name)] = fitted
            timings[col] = {"tune_seconds": round(tune_s, 1), "mean_fit_seconds": round(fit_s, 2)}
            thr = best_f1_threshold(oof.loc[oof.season.isin(cv_seasons), target],
                                    oof.loc[oof.season.isin(cv_seasons), col]) if kind == "clf" else None
            for group, mask in (("cv_2017_2021", oof.season.isin(cv_seasons)),
                                ("holdout_2022", oof.season.isin(split["validation_years"])),
                                ("pooled_2017_2022", oof.season.notna())):
                g = oof[mask]
                m = classification_metrics(g[target], g[col], thr) if kind == "clf" \
                    else regression_metrics(g[target], g[col])
                rm, rt = race_eval(g.assign(**{col: g[col]}), task, col)
                if group == "pooled_2017_2022":
                    race_tables[col] = rt
                rows.append({"task": task, "model": name, "role": role, "group": group,
                             "params": json.dumps(best, default=str), **m, **rm, **timings[col]})
            log(f"{task:8s} {name:22s} " + " ".join(
                f"{k}={rows[-1][k]:.4f}" for k in ("pr_auc", "mae", "winner_top1", "podium_overlap")
                if k in rows[-1] and pd.notna(rows[-1][k])))

    comp = pd.DataFrame(rows)
    # ---- selection (pre-declared) ----
    pooled = comp[(comp.group == "pooled_2017_2022") & (comp.role == "model")]
    for task, (metric, higher) in PRIMARY.items():
        t = pooled[pooled.task == task].sort_values(metric, ascending=not higher)
        selected[task] = t.iloc[0]["model"]
    log(f"selected: {selected}")

    # ---- paired bootstrap: selected vs baselines and runner-up (race level, pooled) ----
    pair_metric = {"winner": "winner_logloss", "podium": "podium_overlap", "position": "rank_mae"}
    pairs = []
    for task, sel in selected.items():
        a = race_tables[f"{task}__{sel}"]
        for other in comp[(comp.task == task)].model.unique():
            if other == sel:
                continue
            b = race_tables[f"{task}__{other}"]
            mcol = pair_metric[task]
            keep = a["winner_metric"] if task == "winner" else np.ones(len(a), bool)
            diff, lo, hi = paired_bootstrap_diff(a.loc[keep, mcol], b.loc[keep, mcol])
            pairs.append({"task": task, "selected": sel, "compared_to": other, "race_metric": mcol,
                          "mean_diff_selected_minus_other": diff, "ci95_low": lo, "ci95_high": hi,
                          "significant": not (lo <= 0 <= hi)})
    pairs = pd.DataFrame(pairs)

    # ---- race-level coherence: tau and podium method (E4) ----
    cvm = oof.season.isin(cv_seasons)
    wcol, pcol, rcol = (f"{t}__{selected[t]}" for t in ("winner", "podium", "position"))
    tau = fit_tau(oof.loc[cvm, wcol], oof.loc[cvm, "race_id"], oof.loc[cvm, "target_winner"])
    oof["win_prob"] = normalize_win(oof[wcol], oof["race_id"], tau)
    oof["podium_prob"] = normalize_podium(oof[pcol], oof["race_id"])
    oof["pred_position"] = oof[rcol]
    hold = oof.season.isin(split["validation_years"])
    tau_check = {
        "tau": tau,
        "holdout_winner_logloss_tau1": float(-np.log(normalize_win(oof.loc[hold, wcol], oof.loc[hold, "race_id"])[
            oof.loc[hold, "target_winner"].to_numpy() == 1]).mean()),
        "holdout_winner_logloss_tau": float(-np.log(oof.loc[hold & (oof.target_winner == 1), "win_prob"]).mean()),
    }
    e4 = []
    for method, score in (("regression", oof["pred_position"]), ("podium_classifier", -oof["podium_prob"]),
                          ("winner_classifier", -oof["win_prob"])):
        d = oof.assign(score=score)
        d["pred_rank"] = final_order(d, "score")
        s = summarize_races(race_table(d))
        e4.append({"method": method, "podium_overlap": s["podium_overlap"], "exact_podium": s["exact_podium"],
                   "mean_spearman": s["mean_spearman"], "rank_mae": s["rank_mae"]})
    e4 = pd.DataFrame(e4)
    podium_method = e4.sort_values(["podium_overlap", "mean_spearman"], ascending=False).iloc[0]["method"]
    log(f"tau={tau:.3f}  podium method={podium_method}")

    # ---- final system on walk-forward predictions ----
    from src.predict import assemble_order   # single definition used by inference too
    oof["pred_rank"] = assemble_order(oof, podium_method)
    sys_rt = race_table(oof)
    system = summarize_races(sys_rt)
    system_ci = {m: bootstrap_ci(sys_rt, m) for m in ("winner_top1", "winner_top3", "podium_overlap",
                                                     "spearman", "rank_mae")}
    hold_sys = summarize_races(sys_rt[sys_rt.race_id.isin(oof.loc[hold, "race_id"])])

    # ---- save walk-forward pipelines of the selected models (replay) and final <=2022 models ----
    for S in wf_seasons:
        d = models_dir / "walkforward" / str(S)
        d.mkdir(parents=True, exist_ok=True)
        for task, sel in selected.items():
            joblib.dump(fitted_all[(task, sel)][S], d / f"{task}.joblib", compress=3)
    final_dir = models_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for task, sel in selected.items():
        target, kind = TASKS[task]
        spec = (classifier_specs if kind == "clf" else regressor_specs)(FEATURES)[sel][0]
        best = json.loads(comp[(comp.task == task) & (comp.model == sel)].iloc[0]["params"])
        m = clone(spec).set_params(**best).fit(X, df[target])
        joblib.dump(m, final_dir / f"{task}.joblib", compress=3)

    meta = {
        "version": "v1", "features": FEATURES, "selected_models": selected,
        "selected_params": {t: json.loads(comp[(comp.task == t) & (comp.model == s)].iloc[0]["params"])
                            for t, s in selected.items()},
        "selection_rule": "best pooled walk-forward 2017-2022 primary metric among ML models "
                          "(winner/podium PR-AUC, position MAE)",
        "tau": tau, "tau_check": tau_check, "podium_method": podium_method,
        "train_start": start, "walk_forward_seasons": wf_seasons, "cv_seasons": cv_seasons,
        "final_models_trained_on": f"{start}-{max(split['validation_years'])}",
        "test_seasons_locked": split["test_years"], "test_evaluated": False,
        "system_walkforward_2017_2022": system, "system_walkforward_ci95": system_ci,
        "system_holdout_2022": hold_sys,
        "seed": SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (models_dir / "metadata_v1.json").write_text(json.dumps(meta, indent=2, default=float))
    comp.to_csv(out_tables / "model_comparison.csv", index=False)
    pd.concat(tuning).to_csv(out_tables / "tuning_results.csv", index=False)
    pairs.to_csv(out_tables / "model_paired_bootstrap.csv", index=False)
    e4.to_csv(out_tables / "e4_podium_method.csv", index=False)
    oof.to_csv(out_pred / "walkforward_predictions.csv", index=False)
    log(f"system (2017-2022 walk-forward): {system}")
    log(f"system holdout 2022: {hold_sys}")


if __name__ == "__main__":
    main()
