"""
Phase 6: experimental analysis on development seasons only (test seasons never loaded).

Tables (outputs/tables/):
    feature_importance.csv   permutation importance of the selected models on hold-out 2022
    calibration_curve.csv    reliability bins of race-normalised win / podium probabilities (2017-2022)
    ablation_groups.csv      E7 drop-one-feature-group + E6 weather variants (train <=2021, eval 2022)
    e11_numpy_vs_sklearn.csv from-scratch gradient-descent logistic regression vs scikit-learn
    season_metrics.csv       final system race metrics per walk-forward season
Figures (outputs/figures/models/*.png), one per question.

Run:  python -m src.analysis   (after python -m src.train)
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, mean_absolute_error, precision_recall_curve  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.evaluate import classification_metrics, race_table, summarize_races  # noqa: E402
from src.features import FEATURE_GROUPS, FEATURES, ORACLE_WEATHER, load_features  # noqa: E402
from src.models import NumpyLogisticRegression, _pipe, classifier_specs, regressor_specs  # noqa: E402
from src.splits import development_rows  # noqa: E402

TASKS = {"winner": ("target_winner", "clf"), "podium": ("target_podium", "clf"),
         "position": ("target_finish_position", "reg")}
BLUE, GRAY, ORANGE = "#2a78d6", "#898781", "#eb6834"


def _score(model, X, y, kind):
    if kind == "clf":
        return average_precision_score(y, model.predict_proba(X)[:, 1])
    return mean_absolute_error(y, model.predict(X))


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)


def main():
    cfg = load_config()
    tables = PROJECT_ROOT / cfg["paths"]["tables"]
    figs = PROJECT_ROOT / cfg["paths"]["figures"] / "models"
    figs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROJECT_ROOT / "models/metadata_v1.json").read_text())
    start = meta["train_start"]
    df = development_rows(load_features())
    df = df[df["season"] >= start]
    tr, va = df[df["season"] <= 2021], df[df["season"] == 2022]
    oof = pd.read_csv(PROJECT_ROOT / cfg["paths"]["predictions"] / "walkforward_predictions.csv")
    comp = pd.read_csv(tables / "model_comparison.csv")

    # ---------- permutation importance (selected 2022 walk-forward models) ----------
    import joblib
    fi_rows = []
    for task, (target, kind) in TASKS.items():
        m = joblib.load(PROJECT_ROOT / "models/walkforward/2022" / f"{task}.joblib")
        r = permutation_importance(m, va[FEATURES], va[target], n_repeats=5, random_state=42, n_jobs=-1,
                                   scoring="average_precision" if kind == "clf" else "neg_mean_absolute_error")
        for f, mu, sd in zip(FEATURES, r.importances_mean, r.importances_std):
            fi_rows.append({"task": task, "model": meta["selected_models"][task], "feature": f,
                            "group": next(g for g, c in FEATURE_GROUPS.items() if f in c),
                            "importance_mean": mu, "importance_std": sd})
    fi = pd.DataFrame(fi_rows).sort_values(["task", "importance_mean"], ascending=[True, False])
    fi.to_csv(tables / "feature_importance.csv", index=False)

    # ---------- calibration (race-normalised probabilities of the final system) ----------
    cal = []
    for task, col, y in (("winner", "win_prob", "target_winner"), ("podium", "podium_prob", "target_podium")):
        g = oof.groupby(pd.qcut(oof[col], 10, duplicates="drop"), observed=True)   # equal-count bins
        b = pd.DataFrame({"mean_predicted": g[col].mean(), "observed_rate": g[y].mean(), "n": g.size()})
        cal += [{"task": task, **r} for r in b.to_dict("records")]
        cal.append({"task": task, "mean_predicted": np.nan, "observed_rate": np.nan, "n": np.nan,
                    "brier": classification_metrics(oof[y], oof[col])["brier"]})
    cal = pd.DataFrame(cal)
    cal[cal["mean_predicted"].notna()].drop(columns="brier").to_csv(tables / "calibration_curve.csv", index=False)

    # ---------- E7 group ablation + E6 weather variants (train <=2021, eval 2022) ----------
    abl = []
    for task, (target, kind) in TASKS.items():
        sel = meta["selected_models"][task]
        specs = (classifier_specs if kind == "clf" else regressor_specs)
        params = meta["selected_params"][task]

        def fit_eval(feats):
            pipe = specs(feats)[sel][0]
            m = clone(pipe).set_params(**params).fit(tr[feats], tr[target])
            return _score(m, va[feats], va[target], kind)

        full = fit_eval(FEATURES)
        abl.append({"task": task, "dropped_group": "none (all features)", "metric": full, "delta": 0.0})
        variants = {g: [f for f in FEATURES if f not in cols] for g, cols in FEATURE_GROUPS.items()}
        variants["E6: race-window weather (oracle) instead of pre-start"] = (
            [f for f in FEATURES if f not in FEATURE_GROUPS["weather_prestart"]] + ORACLE_WEATHER + ["wx_time_imputed"])
        for g, feats in variants.items():
            s = fit_eval(feats)
            abl.append({"task": task, "dropped_group": g, "metric": s, "delta": s - full})
        print(f"ablation {task} done")
    abl = pd.DataFrame(abl)
    abl["metric_name"] = abl["task"].map({"winner": "pr_auc", "podium": "pr_auc", "position": "mae"})
    abl.to_csv(tables / "ablation_groups.csv", index=False)

    # ---------- E11: NumPy gradient descent vs scikit-learn (podium, same preprocessing) ----------
    prep = _pipe(FEATURES, LogisticRegression(), scale=True).named_steps["prep"]
    Xtr = prep.fit_transform(tr[FEATURES])
    Xva = prep.transform(va[FEATURES])
    ours = NumpyLogisticRegression(lr=0.5, n_iter=30000, l2=1e-4, tol=1e-12).fit(Xtr, tr["target_podium"])
    sk = LogisticRegression(C=1 / (1e-4 * len(Xtr)), max_iter=5000).fit(Xtr, tr["target_podium"])
    e11 = []
    for nm, m in (("numpy_gradient_descent", ours), ("sklearn_lbfgs", sk)):
        e11.append({"model": nm, **classification_metrics(va["target_podium"], m.predict_proba(Xva)[:, 1]),
                    "iterations": len(ours.loss_curve_) if nm.startswith("numpy") else int(sk.n_iter_[0])})
    e11 = pd.DataFrame(e11)
    e11["max_abs_coef_diff"] = float(np.max(np.abs(ours.coef_ - sk.coef_[0])))
    e11.to_csv(tables / "e11_numpy_vs_sklearn.csv", index=False)

    # ---------- per-season system metrics ----------
    rt = race_table(oof)
    rt["season"] = rt["race_id"].str[:4].astype(int)
    sm = pd.DataFrame([{"season": s, **summarize_races(g)} for s, g in rt.groupby("season")])
    sm.to_csv(tables / "season_metrics.csv", index=False)

    # ================= figures =================
    pooled = comp[comp["group"] == "pooled_2017_2022"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (task, metric, lbl) in zip(axes, (("winner", "pr_auc", "PR-AUC (higher better)"),
                                              ("podium", "pr_auc", "PR-AUC (higher better)"),
                                              ("position", "mae", "MAE in places (lower better)"))):
        t = pooled[pooled["task"] == task].sort_values(metric, ascending=task != "position")
        colors = [BLUE if m == meta["selected_models"][task] else GRAY if r == "baseline" else "#86b6ef"
                  for m, r in zip(t["model"], t["role"])]
        ax.barh(t["model"].str.replace("_", " "), t[metric], color=colors, height=0.7)
        for y, v in enumerate(t[metric]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color="#52514e")
        _style(ax, f"{task}: {lbl}")
    fig.suptitle("Algorithm comparison, walk-forward 2017-2022 (blue = selected, gray = baseline)", x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figs / "algorithm_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, task, y in ((axes[0], "winner", "target_winner"), (axes[1], "podium", "target_podium")):
        for name, color in ((meta["selected_models"][task], BLUE), ("logistic_regression", ORANGE),
                            ("grid_heuristic", GRAY)):
            col = f"{task}__{name}"
            p, r, _ = precision_recall_curve(oof[y], oof[col])
            ax.plot(r, p, color=color, lw=2, label=f"{name.replace('_', ' ')} (AP {average_precision_score(oof[y], oof[col]):.3f})")
        _style(ax, f"Precision-recall: {task} (2017-2022)", "Recall", "Precision")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "precision_recall.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, task in zip(axes, ("winner", "podium")):
        c = cal[(cal["task"] == task) & cal["mean_predicted"].notna()]
        ax.plot([0, 1], [0, 1], ls=":", color=GRAY, label="perfect calibration")
        ax.plot(c["mean_predicted"], c["observed_rate"], "o-", color=BLUE, lw=2, label="final system")
        _style(ax, f"Calibration: race-normalised {task} probability", "Predicted", "Observed frequency")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "calibration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, task in zip(axes, TASKS):
        t = fi[fi["task"] == task].head(12).iloc[::-1]
        ax.barh(t["feature"], t["importance_mean"], xerr=t["importance_std"], color=BLUE, ecolor=GRAY, height=0.7)
        _style(ax, f"{task} ({meta['selected_models'][task]})", "importance (hold-out 2022)")
    fig.suptitle("Permutation importance of the selected models", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ct = pd.crosstab(oof["pred_rank"].clip(upper=20), oof["target_finish_position"].clip(upper=20), normalize="index")
    im = ax.imshow(ct.to_numpy(), cmap="Blues", origin="lower", aspect="auto")
    ax.set_xticks(range(0, 20, 2), [str(int(v)) for v in ct.columns[::2]])
    ax.set_yticks(range(0, 20, 2), [str(int(v)) for v in ct.index[::2]])
    _style(ax, "Actual finish given predicted position (row share)", "Actual position", "Predicted position")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(figs / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    reg = pooled[pooled["task"] == "position"].sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(reg))
    ax.bar(x - 0.2, reg["mae"], 0.4, color=BLUE, label="MAE")
    ax.bar(x + 0.2, reg["rmse"], 0.4, color=ORANGE, label="RMSE")
    ax.set_xticks(x, reg["model"].str.replace("_", "\n"), fontsize=8)
    _style(ax, "Finishing-position error by regression model (2017-2022)", None, "places")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "regression_errors.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, task in zip(axes, TASKS):
        t = abl[(abl["task"] == task) & (abl["dropped_group"] != "none (all features)")].sort_values("delta")
        good = t["delta"] > 0 if task == "position" else t["delta"] < 0   # metric got worse without the group
        ax.barh(t["dropped_group"].str.slice(0, 40), t["delta"], color=np.where(good, BLUE, GRAY), height=0.7)
        ax.axvline(0, color="#52514e", lw=0.8)
        _style(ax, f"{task}: change in {'MAE' if task == 'position' else 'PR-AUC'} when dropped (2022)")
    fig.tight_layout()
    fig.savefig(figs / "ablation.png", dpi=150)
    plt.close(fig)
    print("analysis tables and figures written")


if __name__ == "__main__":
    main()
