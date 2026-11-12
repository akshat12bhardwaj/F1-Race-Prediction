"""
Course-handout algorithms for the custom race simulator (D-022).

Every supervised algorithm learns from the SAME car-neutral scenario features (ranking.CARNEUTRAL_GROUPS,
incl. grid terms) on the same chronological data, and turns one driver-row into a SCORE:
  - classification algorithms predict P(podium); score = logit(p)   (SVM: its decision function)
  - regression algorithms predict the finishing percentile;  score = -prediction
  - the Plackett-Luce ranking model scores a utility directly (maximum likelihood over finishing orders)
A race is then simulated from scores with one race-level temperature per algorithm, tau, fitted on validation
seasons (validation-based calibration): P(order) = Plackett-Luce with strengths tau * score.

Clustering (K-Means, DBSCAN, BIRCH) and density estimation are NOT race predictors here; see src/unsupervised.py.
"""

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, BaggingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.ranking import CARNEUTRAL_GROUPS, WEATHER_FEATURES

SEED = 42
# D-023: two extra leakage-safe circuit features (expanding means over previous races, features.py)
CIRCUIT_EXTRA = ["circuit_dnf_rate", "constructor_circuit_avg_grid_pct"]
GROUPS_V1 = {**CARNEUTRAL_GROUPS, "circuit_history": CARNEUTRAL_GROUPS["circuit_history"] + CIRCUIT_EXTRA}
FEATURES_V1 = [f for g in GROUPS_V1.values() for f in g]
# D-033: + pre-start weather (rain intensity, air temperature, humidity, wind speed) and pre-registered interactions
GROUPS = {**GROUPS_V1, "weather": WEATHER_FEATURES}
FEATURES_V2 = FEATURES_V1 + WEATHER_FEATURES
FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2}
FEATURES = FEATURES_V1          # reference schema; each saved bundle carries its own "features" list
FEATURE_GROUP = {f: g for g, cols in GROUPS.items() for f in cols}


def _pipe(est, scale=True, impute=True):
    steps = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=True)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("model", est)])


def _tree():
    return DecisionTreeClassifier(random_state=SEED)


# key -> spec. "grid": small hyper-parameter grid chosen on the calibration seasons (see train_algorithms).
REGISTRY = {
    # ---------------- classification (target: podium) ----------------
    "logistic_regression": dict(
        label="Logistic Regression", family="classification", topic="Supervised: classification",
        build=lambda C=1.0: _pipe(LogisticRegression(C=C, max_iter=3000)), grid={"C": [0.1, 1.0]},
        explain="coefficients",
        how="Adds up the features with learned weights (positive weights push towards a podium), then squashes the "
            "total through an S-shaped curve to get a probability.",
        strengths="Simple, fast, transparent: every weight can be read directly. Rarely overfits.",
        weaknesses="Assumes each feature pushes in one direction at a constant rate; cannot learn interactions "
                   "(e.g. 'a strong car matters more at Monaco') unless they are given as features.",
        sensitivity="Changes smoothly with every input: a better car or grid slot always moves the score the same "
                    "way, by an amount set by its weight."),
    "decision_tree": dict(
        label="Decision Tree", family="classification", topic="Supervised: classification",
        build=lambda max_depth=6, min_samples_leaf=50: _pipe(DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED), scale=False),
        grid={"max_depth": [4, 6], "min_samples_leaf": [50]}, explain="tree_path",
        how="Asks a sequence of yes/no questions about the features (e.g. 'grid slot ≤ 2.5?', 'car points share > 0.6?') "
            "and predicts the podium rate of past drivers who ended in the same leaf.",
        strengths="Easy to read as rules; captures thresholds and interactions automatically.",
        weaknesses="High variance: small changes in data can change the tree. Predictions jump in steps, so many "
                   "drivers can receive exactly the same score.",
        sensitivity="Responds only when a change crosses a split threshold; small car or form changes may have no effect."),
    "knn": dict(
        label="K-Nearest Neighbours", family="classification", topic="Supervised: classification",
        build=lambda n_neighbors=101: _pipe(KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
        grid={"n_neighbors": [51, 151]}, explain="neighbours",
        how="Finds the most similar historical driver-races (closest in standardised feature space) and predicts the "
            "share of those neighbours who reached the podium.",
        strengths="No assumptions about the shape of the relationship; explanations are concrete past examples.",
        weaknesses="Struggles with many features and rare outcomes; every feature counts equally after scaling, "
                   "even unimportant ones. Slow on large data.",
        sensitivity="Changes when the scenario moves closer to a different group of past races; can be jumpy."),
    "svm": dict(
        label="Support Vector Machine (RBF)", family="classification", topic="Supervised: classification",
        build=lambda C=1.0, class_weight=None: _pipe(SVC(C=C, kernel="rbf", gamma="scale", class_weight=class_weight)),
        grid={"C": [0.3, 1.0], "class_weight": [None, "balanced"]}, explain="permutation", train_start=1995,
        how="Learns a flexible boundary (via an RBF kernel) that separates podium from non-podium driver-races with "
            "the widest possible margin. The signed distance from the boundary is the score.",
        strengths="Strong with clear margins; the kernel captures non-linear patterns.",
        weaknesses="No native probabilities (here calibrated at race level); slow to train (trained on 1995+ only); "
                   "hard to interpret.",
        sensitivity="Reacts most to changes that move a driver across the boundary region; far-away cases saturate."),
    "naive_bayes": dict(
        label="Gaussian Naive Bayes", family="classification", topic="Bayesian learning",
        build=lambda: _pipe(GaussianNB()), grid={}, explain="class_means",
        how="Applies Bayes' rule: estimates, for podium and non-podium drivers, a bell-curve distribution of every "
            "feature (maximum likelihood), assumes the features are independent, and combines them.",
        strengths="Very fast, needs little data, probabilistic by construction.",
        weaknesses="The independence assumption is badly violated here (grid, car pace and driver skill are "
                   "correlated), so it double-counts evidence and is often over-confident.",
        sensitivity="Every feature shifts the odds independently; correlated features push in the same direction."),
    "random_forest": dict(
        label="Random Forest (decision forest)", family="classification", topic="Ensemble: decision forest",
        build=lambda min_samples_leaf=20: _pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=min_samples_leaf, max_features="sqrt", n_jobs=-1, random_state=SEED),
            scale=False), grid={"min_samples_leaf": [20, 50]}, explain="importances",
        how="Grows hundreds of decision trees, each on a random sample of races and a random subset of features, and "
            "averages their votes.",
        strengths="Much more stable than one tree; captures interactions; robust to irrelevant features.",
        weaknesses="Less transparent; probabilities are averaged votes and tend to be squeezed towards the middle.",
        sensitivity="Smooths out the step changes of a single tree; reacts to combinations of features."),
    "bagging": dict(
        label="Bagging (trees)", family="classification", topic="Ensemble: bagging",
        build=lambda min_samples_leaf=20: _pipe(BaggingClassifier(
            DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=SEED), n_estimators=200,
            n_jobs=-1, random_state=SEED), scale=False), grid={"min_samples_leaf": [20, 50]}, explain="importances",
        how="Trains many decision trees on bootstrap resamples of the training races (all features each time) and "
            "averages them.",
        strengths="Reduces the variance of a single tree.",
        weaknesses="Trees stay similar because every tree sees the same strong features (e.g. grid), so it reduces "
                   "variance less than a random forest.",
        sensitivity="Like a smoothed decision tree."),
    "adaboost": dict(
        label="AdaBoost (boosting)", family="classification", topic="Ensemble: boosting",
        build=lambda n_estimators=200: _pipe(AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2, random_state=SEED), n_estimators=n_estimators, learning_rate=0.1,
            random_state=SEED), scale=False), grid={"n_estimators": [100, 300]}, explain="importances",
        how="Builds small trees one after another; each new tree focuses on the driver-races the previous trees got "
            "wrong (they receive more weight), and the trees vote with learned weights.",
        strengths="Turns weak rules into a strong classifier; good at the hard cases.",
        weaknesses="Sensitive to noisy labels (retirements, crashes) because it keeps up-weighting them.",
        sensitivity="Combines many small threshold rules; reacts to several features at once."),
    "gradient_boosting": dict(
        label="Gradient Boosting", family="classification", topic="Ensemble: boosting",
        build=lambda learning_rate=0.05: _pipe(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=learning_rate, max_leaf_nodes=15, min_samples_leaf=50, early_stopping=True,
            random_state=SEED), scale=False, impute=False), grid={"learning_rate": [0.05, 0.1]},
        explain="permutation",
        how="Adds trees one at a time, each fitted to the remaining errors (the gradient of the loss) of the model so far.",
        strengths="Usually among the most accurate methods on tabular data; handles missing values natively.",
        weaknesses="Many settings to tune; can overfit; hard to interpret.",
        sensitivity="Captures thresholds and interactions; reacts to combinations of driver, car and circuit features."),
    "neural_network": dict(
        label="Neural Network (MLP)", family="classification", topic="Introduction to ANNs",
        build=lambda alpha=1e-3: _pipe(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=alpha, max_iter=400,
                                                     early_stopping=True, random_state=SEED)),
        grid={"alpha": [1e-3, 1e-1]}, explain="permutation",
        how="Passes the features through two layers of artificial neurons (weighted sums + non-linear activations), "
            "trained by gradient descent on the log-loss.",
        strengths="Can learn complex non-linear combinations.",
        weaknesses="Needs more data and tuning; a black box; results depend on initialisation.",
        sensitivity="Smooth but non-linear responses; hard to predict without trying."),
    # ---------------- regression (target: finishing percentile) ----------------
    "linear_regression": dict(
        label="Linear Regression", family="regression", topic="Supervised: regression",
        build=lambda: _pipe(LinearRegression()), grid={}, explain="coefficients",
        how="Predicts the finishing percentile (0 = winner, 1 = last) as a weighted sum of the features, choosing the "
            "weights that minimise the squared error.",
        strengths="Transparent and fast; a strong baseline.",
        weaknesses="Straight-line relationships only; correlated features make individual weights unstable.",
        sensitivity="Every change moves the prediction by weight × change."),
    "ridge": dict(
        label="Ridge Regression", family="regression", topic="Regularisation",
        build=lambda alpha=10.0: _pipe(Ridge(alpha=alpha)), grid={"alpha": [1.0, 100.0]}, explain="coefficients",
        how="Linear regression with an L2 penalty that shrinks all weights towards zero.",
        strengths="Stable when features are correlated (as driver and car measures are).",
        weaknesses="Still linear; keeps every feature.",
        sensitivity="Like linear regression, with smaller and steadier weights."),
    "lasso": dict(
        label="Lasso Regression", family="regression", topic="Regularisation",
        build=lambda alpha=1e-3: _pipe(Lasso(alpha=alpha, max_iter=10000)), grid={"alpha": [1e-4, 3e-3]},
        explain="coefficients",
        how="Linear regression with an L1 penalty that can set weights exactly to zero, keeping only useful features.",
        strengths="Built-in feature selection; easy to read.",
        weaknesses="Among correlated features it may keep one arbitrarily and drop the rest.",
        sensitivity="Ignores features whose weight is zero."),
    "decision_tree_regression": dict(
        label="Decision Tree Regression", family="regression", topic="Supervised: regression",
        build=lambda max_depth=6: _pipe(DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=50,
                                                               random_state=SEED), scale=False),
        grid={"max_depth": [4, 8]}, explain="tree_path",
        how="Splits the data with yes/no questions and predicts the average finishing percentile in each leaf.",
        strengths="Readable rules; captures thresholds.",
        weaknesses="Step-shaped predictions; high variance.",
        sensitivity="Only changes when a split threshold is crossed."),
    "random_forest_regression": dict(
        label="Random Forest Regression", family="regression", topic="Ensemble: decision forest",
        build=lambda min_samples_leaf=20: _pipe(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=min_samples_leaf, max_features=0.5, n_jobs=-1, random_state=SEED),
            scale=False), grid={"min_samples_leaf": [20, 50]}, explain="importances",
        how="Averages hundreds of regression trees grown on random samples of races and features.",
        strengths="Stable, captures interactions.",
        weaknesses="Averaging compresses extreme predictions towards the middle.",
        sensitivity="Smooth, combination-aware responses."),
    "gradient_boosting_regression": dict(
        label="Gradient Boosting Regression", family="regression", topic="Ensemble: boosting",
        build=lambda learning_rate=0.05: _pipe(HistGradientBoostingRegressor(
            max_iter=300, learning_rate=learning_rate, max_leaf_nodes=15, min_samples_leaf=50, early_stopping=True,
            random_state=SEED), scale=False, impute=False), grid={"learning_rate": [0.05, 0.1]},
        explain="permutation",
        how="Adds regression trees one after another, each correcting the remaining errors.",
        strengths="Accurate on tabular data.",
        weaknesses="Needs tuning; less transparent.",
        sensitivity="Captures thresholds and interactions."),
}
# Ranking model (existing simulator default) and a baseline are handled by train_algorithms/simulate.
RANKING_INFO = dict(
    label="Plackett–Luce ranking (simulator default)", family="ranking", topic="Maximum likelihood (Bayesian learning)",
    explain="coefficients",
    how="Learns one strength per car by maximum likelihood over real finishing orders: the winner is picked with "
        "probability proportional to exp(strength), then the next finisher from the rest, and so on.",
    strengths="Models the whole race at once, so probabilities are coherent for any field; transparent weights.",
    weaknesses="Linear strengths; assumes the odds between two drivers do not depend on who else races.",
    sensitivity="Every input shifts the strength by weight × change.")
BASELINE_INFO = dict(
    label="Grid-order baseline", family="baseline", topic="Baseline", explain="none",
    how="Uses only the starting position: a car further up the grid is more likely to finish ahead (Plackett–Luce "
        "on grid terms only).",
    strengths="The bar every real model must beat.",
    weaknesses="Ignores driver, car, circuit and weather.",
    sensitivity="Only the estimated grid slot matters.")


def score(model, family, X):
    """Algorithm score for feature matrix X (higher = better finish)."""
    if family == "classification":
        est = model.steps[-1][1]
        if isinstance(est, SVC):
            return model.decision_function(X)
        if isinstance(est, GaussianNB):     # its probabilities saturate at 0/1; the log-odds ratio does not
            lp = model.predict_log_proba(X)
            return lp[:, 1] - lp[:, 0]
        p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return -model.predict(X)
