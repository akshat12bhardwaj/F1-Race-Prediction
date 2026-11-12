"""
Model ladder (blueprint §23-26): baselines, sklearn models, and a NumPy logistic regression.

Every model is a full Pipeline (column selection -> imputation -> scaling -> estimator) so
preprocessing is fitted only on the training fold.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42
GRID_ONLY = ["grid_position", "grid_pct"]


# ---------------- baselines ----------------
class GridRateClassifier(ClassifierMixin, BaseEstimator):
    """Heuristic: P(target | grid slot) from training data (Laplace-smoothed).
    Race-level argmax = pole sitter (pole-sitter winner heuristic); top-3 = grid 1-3."""

    def __init__(self, max_slot=30, alpha=1.0):
        self.max_slot = max_slot
        self.alpha = alpha

    def _slot(self, X):
        g = np.asarray(X["grid_position"], float)
        return np.where(np.isnan(g), self.max_slot + 1, np.clip(g, 1, self.max_slot)).astype(int)

    def fit(self, X, y):
        s, y = self._slot(X), np.asarray(y, float)
        prior = y.mean()
        n = np.bincount(s, minlength=self.max_slot + 2)
        k = np.bincount(s, weights=y, minlength=self.max_slot + 2)
        self.rate_ = (k + self.alpha * prior) / (n + self.alpha)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.rate_[self._slot(X)]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GridOrderRegressor(RegressorMixin, BaseEstimator):
    """Heuristic: predicted finishing position = grid position (unknown grid -> back of field)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        g = np.asarray(X["grid_position"], float)
        return np.where(np.isnan(g), np.asarray(X["field_size"], float), g)


# ---------------- from scratch ----------------
class NumpyLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression trained with full-batch gradient descent on the
    L2-regularised mean log-loss:  J(w,b) = -mean[y log s + (1-y) log(1-s)] + (l2/2)||w||^2,
    gradient dJ/dw = X^T (s - y)/n + l2 w,  dJ/db = mean(s - y).  Inputs must be scaled."""

    def __init__(self, lr=0.5, n_iter=2000, l2=1e-4, tol=1e-7):
        self.lr, self.n_iter, self.l2, self.tol = lr, n_iter, l2, tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        self.loss_curve_ = []
        for _ in range(self.n_iter):
            s = 1 / (1 + np.exp(-(X @ w + b)))
            err = s - y
            w -= self.lr * (X.T @ err / n + self.l2 * w)
            b -= self.lr * err.mean()
            s = np.clip(s, 1e-12, 1 - 1e-12)
            self.loss_curve_.append(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)) + self.l2 / 2 * w @ w)
            if len(self.loss_curve_) > 1 and abs(self.loss_curve_[-2] - self.loss_curve_[-1]) < self.tol:
                break
        self.coef_, self.intercept_, self.classes_ = w, b, np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(np.asarray(X, float) @ self.coef_ + self.intercept_)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------- pipelines ----------------
def _prep(features, scale):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), list(features))], remainder="drop")


def _pipe(features, est, scale=False, impute=True):
    if not impute:   # HistGradientBoosting handles NaN natively
        return Pipeline([("select", ColumnTransformer([("num", "passthrough", list(features))])),
                         ("model", est)])
    return Pipeline([("prep", _prep(features, scale)), ("model", est)])


def classifier_specs(features):
    """name -> (pipeline, param space, search kind, role). role: baseline | model."""
    return {
        "dummy_prior": (Pipeline([("model", DummyClassifier(strategy="prior"))]), {}, None, "baseline"),
        "grid_heuristic": (Pipeline([("model", GridRateClassifier())]), {}, None, "baseline"),
        "grid_logistic": (_pipe(GRID_ONLY, LogisticRegression(max_iter=2000), scale=True), {}, None,
                          "baseline"),
        "logistic_regression": (
            _pipe(features, LogisticRegression(max_iter=3000, solver="saga"), scale=True),
            {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2", "l1"],
             "model__class_weight": [None, "balanced"]}, "grid", "model"),
        "numpy_logistic_gd": (_pipe(features, NumpyLogisticRegression(), scale=True), {}, None, "model"),
        "decision_tree": (
            _pipe(features, DecisionTreeClassifier(random_state=SEED)),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 20, 50, 100]}, "grid", "model"),
        "random_forest": (
            _pipe(features, RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
            {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [5, 20, 50],
             "model__max_features": ["sqrt", 0.5], "model__class_weight": [None, "balanced_subsample"]},
            ("random", 8), "model"),
        "gradient_boosting": (
            _pipe(features, HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                                           random_state=SEED), impute=False),
            {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [15, 31],
             "model__min_samples_leaf": [20, 50, 100], "model__l2_regularization": [0.0, 1.0]},
            ("random", 10), "model"),
    }


def regressor_specs(features):
    return {
        "dummy_mean": (Pipeline([("model", DummyRegressor())]), {}, None, "baseline"),
        "grid_heuristic": (Pipeline([("model", GridOrderRegressor())]), {}, None, "baseline"),
        "grid_linear": (_pipe(GRID_ONLY, LinearRegression()), {}, None, "baseline"),
        "linear_regression": (_pipe(features, LinearRegression(), scale=True), {}, None, "model"),
        "ridge": (_pipe(features, Ridge(), scale=True), {"model__alpha": np.logspace(-3, 3, 13).tolist()},
                  "grid", "model"),
        "lasso": (_pipe(features, Lasso(max_iter=5000), scale=True),
                  {"model__alpha": np.logspace(-4, 0, 9).tolist()}, "grid", "model"),
        "decision_tree": (
            _pipe(features, DecisionTreeRegressor(random_state=SEED)),
            {"model__max_depth": [3, 5, 8, 12], "model__min_samples_leaf": [5, 20, 50, 100]}, "grid", "model"),
        "random_forest": (
            _pipe(features, RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
            {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [5, 20, 50],
             "model__max_features": ["sqrt", 0.5]}, ("random", 8), "model"),
        "gradient_boosting": (
            _pipe(features, HistGradientBoostingRegressor(max_iter=500, early_stopping=True,
                                                          random_state=SEED), impute=False),
            {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [15, 31],
             "model__min_samples_leaf": [20, 50, 100], "model__l2_regularization": [0.0, 1.0]},
            ("random", 10), "model"),
    }
