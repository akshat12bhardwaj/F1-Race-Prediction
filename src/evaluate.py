"""
Evaluation metrics (blueprint §28) and race-level probability coherence (§30.3).

Race-level coherence:
- Win: p_i^tau / sum_j p_j^tau within the race (sums to 1). tau is a single temperature
  fitted on out-of-fold predictions (conditional-logit style calibration).
- Podium: scaled so the race sums to min(3, n), each value capped at 1 (iterative
  water-filling), so exactly three podium places are distributed.
- Final order: unique, deterministic ranks (see `final_order`).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

EPS = 1e-12


# ---------------- race-level probability coherence ----------------
def normalize_win(p, race_ids, tau=1.0):
    p = np.clip(np.asarray(p, float), EPS, 1.0) ** tau
    s = pd.Series(p).groupby(np.asarray(race_ids)).transform("sum").to_numpy()
    return p / s


def scale_capped(p, total):
    """Scale non-negative p so it sums to `total` with every value <= 1 (water-filling)."""
    p = np.clip(np.asarray(p, float), EPS, None)
    out = np.zeros_like(p)
    free = np.ones(len(p), bool)
    remaining = float(min(total, len(p)))
    for _ in range(len(p)):
        scaled = p[free] * remaining / p[free].sum()
        over = scaled > 1
        if not over.any():
            out[free] = scaled
            break
        idx = np.where(free)[0][over]
        out[idx] = 1.0
        free[idx] = False
        remaining -= len(idx)
    return out


def normalize_podium(p, race_ids):
    p = np.asarray(p, float)
    out = np.empty_like(p)
    for _, idx in pd.Series(np.arange(len(p))).groupby(np.asarray(race_ids)):
        out[idx.to_numpy()] = scale_capped(p[idx.to_numpy()], 3)
    return out


def fit_tau(p, race_ids, y_win):
    """Temperature minimising race-level winner log-loss on out-of-fold predictions."""
    y = np.asarray(y_win) == 1

    def loss(t):
        return -np.mean(np.log(normalize_win(p, race_ids, t)[y] + EPS))

    return float(minimize_scalar(loss, bounds=(0.2, 5.0), method="bounded").x)


def final_order(df, score_col, tie_cols=("win_prob", "grid_position"), asc=(True, False, True)):
    """Unique 1..n rank within each race: by score (lower = better), then higher win prob, then
    better grid, then driver_id for full determinism."""
    cols = [score_col, *tie_cols, "driver_id"]
    order = df.sort_values(["race_id", *cols], ascending=[True, *asc, True], na_position="last")
    return order.groupby("race_id").cumcount().add(1).reindex(df.index)


# ---------------- row-level metrics ----------------
def classification_metrics(y, p, threshold=None):
    y, p = np.asarray(y), np.asarray(p)
    out = {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
           "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)), "brier": brier_score_loss(y, p),
           "base_rate": y.mean()}
    if threshold is not None:
        yh = p >= threshold
        out.update(threshold=threshold, precision=precision_score(y, yh, zero_division=0),
                   recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0))
    return out


def best_f1_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 100)))
    return float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))


def regression_metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"mae": mean_absolute_error(y, yhat), "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
            "r2": r2_score(y, yhat), "spearman": spearmanr(y, yhat).statistic}


# ---------------- race-level metrics ----------------
def race_table(df):
    """Per-race outcomes. df needs race_id, driver_id, target_finish_position, target_winner,
    win_prob (race-normalised), podium_prob, pred_rank (unique 1..n), exclude_from_winner_metrics."""
    rows = []
    # Row order must never matter: input files are sorted by actual finish, so every tie is broken
    # by pre-race information only (grid, then driver_id).
    df = df.sort_values(["race_id", "grid_position", "driver_id"], na_position="last", kind="mergesort")
    for rid, g in df.groupby("race_id", sort=False):
        actual_pod = set(g.loc[g["target_finish_position"] <= 3, "driver_id"])
        pred_pod = list(g.nsmallest(3, "pred_rank")["driver_id"])
        by_win = g.sort_values("win_prob", ascending=False, kind="mergesort")["driver_id"].tolist()
        winners = set(g.loc[g["target_winner"] == 1, "driver_id"])
        actual_order = g.sort_values("target_finish_position")["driver_id"].tolist()
        rows.append({
            "race_id": rid,
            "winner_metric": not bool(g["exclude_from_winner_metrics"].iloc[0]),
            "winner_top1": by_win[0] in winners,
            "winner_top3": bool(winners & set(by_win[:3])),
            "winner_logloss": -np.log(g.loc[g["target_winner"] == 1, "win_prob"].max() + EPS),
            "podium_overlap": len(actual_pod & set(pred_pod)) / 3,
            "exact_podium": pred_pod == actual_order[:3],
            "spearman": spearmanr(g["target_finish_position"], g["pred_rank"]).statistic,
            "rank_mae": np.mean(np.abs(g["target_finish_position"] - g["pred_rank"])),
        })
    return pd.DataFrame(rows)


def summarize_races(rt):
    w = rt[rt["winner_metric"]]
    return {"races": len(rt), "winner_top1": w["winner_top1"].mean(), "winner_top3": w["winner_top3"].mean(),
            "winner_logloss": w["winner_logloss"].mean(), "podium_overlap": rt["podium_overlap"].mean(),
            "exact_podium": rt["exact_podium"].mean(), "mean_spearman": rt["spearman"].mean(),
            "rank_mae": rt["rank_mae"].mean()}


def bootstrap_ci(rt, metric, n=1000, seed=42):
    """95% CI by resampling races (not rows)."""
    vals = rt[rt["winner_metric"]][metric] if metric.startswith("winner") else rt[metric]
    vals = vals.astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_bootstrap_diff(a, b, n=1000, seed=42):
    """95% CI of mean(a - b) over races (a, b aligned per race)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
