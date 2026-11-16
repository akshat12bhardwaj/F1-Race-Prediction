"""
Custom-race (scenario) model experiments, selection, one-time final test, production fit (D-019/D-020).

Protocol (pre-declared before results were seen):
1. Development, walk-forward 2017-2022 (train seasons < S, predict S). Candidates: pl_grid (baseline),
   pl_linear (train start 1950/1981, half-life none/8 seasons), hybrid_gbm. The v1 historical system
   (classification + regression + race normalisation) is evaluated on the same races.
   Selection: lowest pooled winner log-loss among non-baseline scenario candidates.
2. Subset experiment (custom fields from real data): for each development race, 3 random subsets
   of 6 starters; target = their actual relative order. Compares PL vs renormalised v1 probabilities.
3. Final test 2023-2025, run ONCE after step 1 is frozen (first access to these outcomes; authorised
   2026-11-07): selected scenario model trained on <= 2022, and the frozen v1 final models.
4. Production: refit the selected configuration on all seasons up to 2025 for the simulator.

Run:  python -m src.train_scenario
"""

import json
import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss

from src.config import PROJECT_ROOT, load_config
from src.evaluate import bootstrap_ci, normalize_podium, normalize_win, race_table, summarize_races
from src.features import FEATURES, load_features
from src.predict import assemble_order
from src.ranking import SCENARIO_FEATURES, ScenarioRanker, race_probabilities

SEED = 42
# Candidates whose selection was frozen before 2023-2025 outcomes were first accessed (run 1, 2026-11-07).
RUN1_CANDIDATES = {"pl_grid", "hybrid_gbm", "pl_linear_from1950", "pl_linear_from1950_hl8",
                   "pl_linear_from1981", "pl_linear_from1981_hl8"}
DEV_SEASONS = [2017, 2018, 2019, 2020, 2021, 2022]
TEST_SEASONS = [2023, 2024, 2025]
N_SIMS = 2000


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def race_outputs(df, u, seed=SEED):
    """Per-row win (exact), podium (MC) and predicted rank (order of decreasing utility)."""
    out = df[["race_id", "season", "driver_id", "grid_position", "target_finish_position", "target_winner",
              "target_podium", "exclude_from_winner_metrics"]].copy()
    out["u"] = u
    win, pod = np.empty(len(df)), np.empty(len(df))
    for _, idx in out.groupby("race_id").indices.items():
        p = race_probabilities(out["u"].to_numpy()[idx], N_SIMS, seed)
        win[idx], pod[idx] = p["win"], p["podium"]
    out["win_prob"], out["podium_prob"] = win, pod
    out["pred_rank"] = out.assign(score=-out["u"]).sort_values(
        ["race_id", "score", "grid_position", "driver_id"], na_position="last").groupby("race_id").cumcount() \
        .add(1).reindex(out.index)
    return out


def metrics(o):
    s = summarize_races(race_table(o))
    return {**s, "win_brier": brier_score_loss(o["target_winner"], o["win_prob"]),
            "win_pr_auc": average_precision_score(o["target_winner"], o["win_prob"]),
            "podium_brier": brier_score_loss(o["target_podium"], o["podium_prob"]),
            "podium_pr_auc": average_precision_score(o["target_podium"], o["podium_prob"])}


def subset_eval(o, k=6, reps=3, seed=SEED):
    """Custom-field check: random k-driver subsets of real races, scored on their actual relative order."""
    rng = np.random.default_rng(seed)
    rows = []
    for rid, g in o.groupby("race_id"):
        for rep in range(reps):
            s = g.iloc[rng.choice(len(g), min(k, len(g)), replace=False)]
            actual = s["target_finish_position"].rank(method="first").to_numpy()
            p = s["sub_win"].to_numpy() / s["sub_win"].sum() if "sub_win" in s else win_from_u(s["u"].to_numpy())
            w = actual == 1
            rows.append({"race_id": rid, "rep": rep, "winner_logloss": -np.log(p[w][0] + 1e-12),
                         "winner_top1": bool(w[np.argmax(p)]),
                         "spearman": spearmanr(-p if "sub_win" in s else -s["u"], actual).statistic})
    return pd.DataFrame(rows)


def win_from_u(u):
    e = np.exp(u - u.max())
    return e / e.sum()


def fit_predict(cfg, train, test):
    m = ScenarioRanker(cfg["kind"], half_life_seasons=cfg.get("half_life"), top_k=cfg.get("top_k")).fit(
        train[train["season"] >= cfg["train_start"]])
    return m, m.utility(test)


def v1_system(df, models, tau, podium_method, coherent=False):
    """Frozen v1 historical system on given rows (same coherence steps as the prediction engine)."""
    X = df[FEATURES]
    o = df[["race_id", "season", "driver_id", "grid_position", "target_finish_position", "target_winner",
            "target_podium", "exclude_from_winner_metrics"]].copy()
    o["win_prob"] = normalize_win(models["winner"].predict_proba(X)[:, 1], o["race_id"], tau)
    o["podium_prob"] = normalize_podium(models["podium"].predict_proba(X)[:, 1], o["race_id"], o["win_prob"].to_numpy() if coherent else None)
    o["pred_position"] = models["position"].predict(X)
    o["pred_rank"] = assemble_order(o, podium_method)
    return o


def main():
    cfg = load_config()
    tables = PROJECT_ROOT / cfg["paths"]["tables"]
    models_dir = PROJECT_ROOT / cfg["paths"]["models"]
    mf = load_features()
    meta_v1 = json.loads((models_dir / "metadata_v1.json").read_text())

    # ---------------- 1. development walk-forward ----------------
    cands = [{"name": "pl_grid", "kind": "pl_grid", "train_start": 1950, "role": "baseline"},
             {"name": "hybrid_gbm", "kind": "hybrid_gbm", "train_start": 1950, "role": "model"}]
    for start, hl in product([1950, 1981], [None, 8]):
        cands.append({"name": f"pl_linear_from{start}" + (f"_hl{hl}" if hl else ""), "kind": "pl_linear",
                      "train_start": start, "half_life": hl, "role": "model"})
    # run 2 (added after run 1 showed poorly calibrated full-order PL win probabilities on 2017-2022):
    # truncated top-k PL (Henderson & Kirrane 2018)
    for k in (1, 3, 5, 10):
        cands.append({"name": f"pl_linear_from1981_hl8_top{k}", "kind": "pl_linear", "train_start": 1981,
                      "half_life": 8, "top_k": k, "role": "model"})
    dev = mf[mf["season"].isin(DEV_SEASONS)]
    outs, rows = {}, []
    for c in cands:
        parts = []
        t0 = time.perf_counter()
        for S in DEV_SEASONS:
            _, u = fit_predict(c, mf[mf["season"] < S], mf[mf["season"] == S])
            parts.append(race_outputs(mf[mf["season"] == S], u))
        o = pd.concat(parts)
        outs[c["name"]] = o
        rows.append({"model": c["name"], "role": c["role"], "kind": c["kind"], "train_start": c["train_start"],
                     "half_life": c.get("half_life"), **metrics(o),
                     "fit_seconds_per_season": (time.perf_counter() - t0) / len(DEV_SEASONS)})
        log(f"{c['name']:28s} winner_ll={rows[-1]['winner_logloss']:.4f} top1={rows[-1]['winner_top1']:.3f} "
            f"podium={rows[-1]['podium_overlap']:.3f} rho={rows[-1]['mean_spearman']:.3f}")
    v1 = pd.read_csv(PROJECT_ROOT / cfg["paths"]["predictions"] / "walkforward_predictions.csv")
    outs["v1_classification_regression"] = v1
    rows.append({"model": "v1_classification_regression", "role": "existing", "kind": "per-driver + normalisation",
                 **metrics(v1)})
    dev_table = pd.DataFrame(rows)

    sel = dev_table[dev_table["role"] == "model"].sort_values("winner_logloss").iloc[0]
    sel_cfg = next(c for c in cands if c["name"] == sel["model"])
    log(f"selected scenario model: {sel_cfg['name']}")

    # paired race bootstrap: selected vs others on winner log-loss and rank MAE
    pairs = []
    a_rt = race_table(outs[sel_cfg["name"]]).set_index("race_id")
    for name, o in outs.items():
        if name == sel_cfg["name"]:
            continue
        b_rt = race_table(o).set_index("race_id").reindex(a_rt.index)
        for m in ("winner_logloss", "rank_mae", "podium_overlap"):
            keep = a_rt["winner_metric"] if m.startswith("winner") else np.ones(len(a_rt), bool)
            d = (a_rt.loc[keep, m].astype(float) - b_rt.loc[keep, m].astype(float)).to_numpy()
            boots = d[np.random.default_rng(SEED).integers(0, len(d), (1000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pairs.append({"selected": sel_cfg["name"], "compared_to": name, "race_metric": m,
                          "mean_diff_selected_minus_other": d.mean(), "ci95_low": lo, "ci95_high": hi,
                          "significant": not (lo <= 0 <= hi)})

    # ---------------- 2. subset (custom-field) experiment ----------------
    sub_rows = []
    for name in (sel_cfg["name"], "pl_grid", "v1_classification_regression"):
        o = outs[name].copy()
        if name == "v1_classification_regression":
            o["sub_win"] = o["win_prob"]          # naive: renormalise the full-field win probabilities
        s = subset_eval(o)
        sub_rows.append({"model": name, "subsets": len(s), "subset_winner_logloss": s["winner_logloss"].mean(),
                         "subset_winner_top1": s["winner_top1"].mean(), "subset_spearman": s["spearman"].mean()})
    subset_table = pd.DataFrame(sub_rows)
    log(f"subset experiment:\n{subset_table.round(4).to_string(index=False)}")

    # ---------------- 3. one-time final test 2023-2025 (frozen choices) ----------------
    clean = sel_cfg["name"] in RUN1_CANDIDATES
    log("TEST 2023-2025: " + ("clean hold-out (candidate frozen in run 1)" if clean else
                              "POST-HOC: 2023-2025 outcomes were already seen in run 1"))
    test = mf[mf["season"].isin(TEST_SEASONS)]
    _, u = fit_predict(sel_cfg, mf[mf["season"] <= 2022], test)
    final_outs = {sel_cfg["name"]: race_outputs(test, u)}
    _, u = fit_predict(cands[0], mf[mf["season"] <= 2022], test)
    final_outs["pl_grid"] = race_outputs(test, u)
    v1_models = {t: joblib.load(models_dir / "final" / f"{t}.joblib") for t in ("winner", "podium", "position")}
    final_outs["v1_classification_regression"] = v1_system(test, v1_models, meta_v1["tau"], meta_v1["podium_method"])
    final_rows = []
    for name, o in final_outs.items():
        rt = race_table(o)
        ci = {f"{m}_ci95": bootstrap_ci(rt, m) for m in ("winner_top1", "winner_logloss", "podium_overlap",
                                                          "spearman", "rank_mae")}
        status = "clean hold-out (v1 frozen 2026-11-06)" if name.startswith("v1") else (
            "clean hold-out" if name in RUN1_CANDIDATES else "post-hoc (test seasons seen in run 1)")
        final_rows.append({"model": name, "evaluation_status": status, **metrics(o),
                           **{k: list(v) for k, v in ci.items()}})
    final_table = pd.DataFrame(final_rows)
    log(f"final test:\n{final_table[['model', 'races', 'winner_top1', 'winner_logloss', 'podium_overlap', 'mean_spearman', 'rank_mae']].round(4).to_string(index=False)}")
    per_season = []
    for name, o in final_outs.items():
        for S, g in o.groupby("season"):
            per_season.append({"model": name, "season": S, **summarize_races(race_table(g))})

    # ---------------- 4. production fit on all seasons ----------------
    prod = ScenarioRanker(sel_cfg["kind"], half_life_seasons=sel_cfg.get("half_life"), top_k=sel_cfg.get("top_k")).fit(
        mf[mf["season"] >= sel_cfg["train_start"]])
    out_dir = models_dir / "scenario"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod, out_dir / "ranker_v1.joblib", compress=3)
    coef = None
    if sel_cfg["kind"] != "hybrid_gbm":
        coef = dict(zip(prod.prep_.get_feature_names_out(), prod.pl_.coef_.round(4)))
    meta = {"version": "scenario_v1", "selected": sel_cfg, "features": SCENARIO_FEATURES,
            "selection_rule": "lowest pooled walk-forward 2017-2022 winner log-loss among scenario candidates",
            "trained_on": f"{sel_cfg['train_start']}-2025", "n_sims_default": 10000, "seed": SEED,
            "coefficients_standardised": coef, "pl_converged": bool(prod.pl_.converged_),
            "development_2017_2022": dev_table.set_index("model").loc[sel_cfg["name"]].to_dict(),
            "final_test_2023_2025": final_table.set_index("model").loc[sel_cfg["name"]].to_dict(),
            "final_test_status": "clean hold-out" if clean else "post-hoc (2023-2025 outcomes seen in run 1)",
            "final_test_note": "2023-2025 outcomes were first accessed in run 1 (2026-11-07) after the run-1 "
                               "selection was frozen. Truncated top-k candidates were added afterwards, so their "
                               "2023-2025 numbers are post-hoc, not a clean hold-out. The production model is "
                               "refit on all seasons up to 2025.",
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=lambda x: None if x != x else float(x)))

    dev_table.to_csv(tables / "scenario_model_comparison.csv", index=False)
    pd.DataFrame(pairs).to_csv(tables / "scenario_paired_bootstrap.csv", index=False)
    subset_table.to_csv(tables / "scenario_subset_experiment.csv", index=False)
    final_table.to_csv(tables / "final_test_2023_2025.csv", index=False)
    pd.DataFrame(per_season).to_csv(tables / "final_test_by_season.csv", index=False)

    # unlock 2023-2025 replay for the v1 historical models (trained <= 2022: out-of-sample for 2023-2025)
    v1_final = final_table.set_index("model").loc["v1_classification_regression"].to_dict()
    meta_v1.update(test_evaluated=True, test_evaluated_on=time.strftime("%Y-%m-%d"),
                   final_test_2023_2025=v1_final)
    (models_dir / "metadata_v1.json").write_text(json.dumps(meta_v1, indent=2, default=lambda x: None if x != x else float(x)))
    log("done")


if __name__ == "__main__":
    main()
