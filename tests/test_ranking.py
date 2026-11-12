"""Plackett-Luce ranking model: math checks on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from src.ranking import PlackettLuce, _pad, pl_loglik_grad, race_probabilities, simulate_orders, win_probs


def _synthetic(n_races=400, n=12, w=(1.5, -0.8, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_races * n, len(w)))
    u = X @ np.array(w)
    races = np.repeat(np.arange(n_races), n)
    finish = np.empty(len(u))
    for r in range(n_races):
        idx = np.where(races == r)[0]
        g = u[idx] + rng.gumbel(size=n)
        finish[idx[np.argsort(-g)]] = np.arange(1, n + 1)
    return X, races, finish


def test_gradient_matches_finite_differences():
    X, races, finish = _synthetic(30, 6)
    Xp, mask, _ = _pad(X, races, finish)
    f = lambda w: pl_loglik_grad(w, Xp, mask, np.ones(len(Xp)), 0.01)[0]
    g = lambda w: pl_loglik_grad(w, Xp, mask, np.ones(len(Xp)), 0.01)[1]
    assert check_grad(f, g, np.array([0.3, -0.2, 0.1])) < 1e-5


@pytest.mark.parametrize("k", [1, 3])
def test_truncated_gradient_matches_finite_differences(k):
    X, races, finish = _synthetic(30, 6)
    Xp, mask, _ = _pad(X, races, finish)
    f = lambda w: pl_loglik_grad(w, Xp, mask, np.ones(len(Xp)), 0.01, k)[0]
    g = lambda w: pl_loglik_grad(w, Xp, mask, np.ones(len(Xp)), 0.01, k)[1]
    assert check_grad(f, g, np.array([0.3, -0.2, 0.1])) < 1e-5


def test_truncated_pl_also_recovers_weights():
    X, races, finish = _synthetic(1500, 10)
    m = PlackettLuce(alpha=0, top_k=3).fit(X, races, finish)
    assert np.allclose(m.coef_, [1.5, -0.8, 0.0], atol=0.15)


def test_recovers_true_weights_with_ragged_races():
    X, races, finish = _synthetic(600, 10)
    keep = ~((races % 3 == 0) & (finish > 7))          # some races shorter (padding path)
    m = PlackettLuce(alpha=0).fit(X[keep], races[keep], finish[keep])
    assert np.allclose(m.coef_, [1.5, -0.8, 0.0], atol=0.12)


def test_luce_subset_property_and_coherence():
    u = np.array([1.2, 0.4, 0.0, -0.5, -1.0, 0.8])
    p = race_probabilities(u, n_sims=200_000, seed=1)
    assert p["win"].sum() == pytest.approx(1) and p["podium"].sum() == pytest.approx(3)
    assert p["top5"].sum() == pytest.approx(5) and p["dist"].sum(axis=0) == pytest.approx(np.ones(6))
    assert np.allclose(p["dist"][:, 0], p["win"], atol=0.005)       # sampled P1 == exact softmax
    # P(a beats b) in the full field equals the 2-driver PL probability (independence of irrelevant alternatives)
    pos = simulate_orders(u, 200_000, seed=2)
    assert (pos[:, 0] < pos[:, 1]).mean() == pytest.approx(win_probs(u[:2])[0], abs=0.005)


def test_simulation_reproducible_and_valid_orders():
    u = np.array([0.5, 0.1, -0.3])
    a, b = simulate_orders(u, 1000, seed=7), simulate_orders(u, 1000, seed=7)
    assert (a == b).all() and (np.sort(a, axis=1) == [1, 2, 3]).all()
    small = race_probabilities(np.array([0.3, -0.3]), 1000)
    assert small["podium"].sum() == pytest.approx(2)              # never more podium places than drivers
