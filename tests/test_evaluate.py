"""Evaluation and race-level coherence helpers."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (final_order, fit_tau, normalize_podium, normalize_win, paired_bootstrap_diff,
                          race_table, scale_capped)
from src.models import GridRateClassifier, NumpyLogisticRegression


def _race(n=6, rid="r1"):
    return pd.DataFrame({"race_id": rid, "driver_id": [f"d{i}" for i in range(n)],
                         "grid_position": np.arange(1, n + 1, dtype=float),
                         "target_finish_position": np.arange(1, n + 1, dtype=float),
                         "target_winner": [1] + [0] * (n - 1), "exclude_from_winner_metrics": False})


def test_tied_probabilities_do_not_leak_row_order():
    """Regression: a constant (dummy) model must not 'win' because rows are sorted by finish."""
    d = _race()
    d["target_finish_position"] = [6, 5, 4, 3, 2, 1.0]       # back of the grid wins
    d["target_winner"] = [0, 0, 0, 0, 0, 1]
    d = d.sort_values("target_finish_position")              # same order as the feature file
    d["win_prob"] = 1 / 6
    d["pred_rank"] = final_order(d.assign(score=0.0), "score")
    rt = race_table(d)
    assert not rt["winner_top1"].iloc[0]
    shuffled = race_table(d.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(rt, shuffled)


def test_win_probs_sum_to_one_and_podium_to_three():
    d = pd.concat([_race(20, "a"), _race(8, "b")])
    p = np.random.default_rng(0).uniform(0.01, 0.9, len(d))
    w = normalize_win(p, d["race_id"], tau=1.3)
    pod = normalize_podium(p, d["race_id"])
    assert np.allclose(pd.Series(w).groupby(d["race_id"].values).sum(), 1)
    assert np.allclose(pd.Series(pod).groupby(d["race_id"].values).sum(), 3)
    assert (pod <= 1 + 1e-12).all() and (pod > 0).all()


def test_scale_capped_caps_at_one():
    out = scale_capped(np.array([0.99, 0.98, 0.97, 0.01]), 3)
    assert out.sum() == pytest.approx(3) and out.max() <= 1


def test_final_order_unique_and_deterministic():
    d = _race(5)
    d["score"] = [2.0, 1.0, 1.0, 3.0, 3.0]
    d["win_prob"] = [0.1, 0.3, 0.3, 0.2, 0.1]
    r1 = final_order(d, "score")
    r2 = final_order(d.sample(frac=1, random_state=3), "score").reindex(d.index)
    assert sorted(r1) == [1, 2, 3, 4, 5] and (r1 == r2).all()
    assert list(r1) == [3, 1, 2, 4, 5]   # tie on score + win prob -> better grid first


def test_fit_tau_recovers_sharpening():
    rng = np.random.default_rng(0)
    races = np.repeat(np.arange(200), 10)
    strength = rng.normal(size=races.size)
    true = normalize_win(np.exp(strength), races, 2.0)
    winners = pd.Series(true).groupby(races).transform(lambda s: s.index == rng.choice(s.index, p=s.values))
    tau = fit_tau(np.exp(strength), races, winners.astype(int))
    assert 1.5 < tau < 2.6


def test_paired_bootstrap_detects_difference():
    a, b = np.ones(100), np.zeros(100)
    d, lo, hi = paired_bootstrap_diff(a, b)
    assert d == 1 and lo == hi == 1


def test_grid_rate_classifier_learns_pole_advantage():
    X = pd.DataFrame({"grid_position": [1, 1, 2, 2, 3, 3, np.nan]})
    y = [1, 1, 0, 0, 0, 0, 0]
    p = GridRateClassifier().fit(X, y).predict_proba(X)[:, 1]
    assert p[0] > p[2] > 0 and p[-1] < p[0]


def test_numpy_logreg_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X @ [1.5, -1, 0.5] + rng.logistic(size=2000) > 0).astype(int)
    ours = NumpyLogisticRegression(lr=0.5, n_iter=5000, l2=0, tol=1e-10).fit(X, y)
    sk = LogisticRegression(penalty=None).fit(X, y)
    assert np.allclose(ours.coef_, sk.coef_[0], atol=0.02)
    assert ours.loss_curve_[-1] < ours.loss_curve_[0]
