"""Course algorithms in the custom race simulator (D-022)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models/algorithms/metadata.json").exists(),
                                reason="algorithms not trained")
SYLLABUS = {"logistic_regression": "classification", "decision_tree": "classification", "knn": "classification",
            "svm": "classification", "random_forest": "classification", "bagging": "classification",
            "adaboost": "classification", "gradient_boosting": "classification", "naive_bayes": "classification",
            "neural_network": "classification", "linear_regression": "regression", "ridge": "regression",
            "lasso": "regression", "decision_tree_regression": "regression", "random_forest_regression": "regression",
            "gradient_boosting_regression": "regression", "plackett_luce": "ranking", "grid_baseline": "baseline"}


@pytest.fixture(scope="module")
def client():
    from src.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def grid(client):
    from conftest import prime_field
    return prime_field("monaco")


@pytest.fixture(scope="module")
def all_runs(client, grid):
    r = client.post("/api/compare-algorithms", json={"circuit_id": "monaco", "entries": grid, "algorithms": ["all"]})
    assert r.status_code == 200, r.text
    return r.json()


def test_catalog_covers_course_algorithms_with_real_metrics(client):
    cat = {a["key"]: a for a in client.get("/api/algorithms").json()}
    assert {k: v["family"] for k, v in cat.items()} == SYLLABUS
    assert sum(a["best_validated"] for a in cat.values()) == 1
    for k, a in cat.items():
        assert a["how"] and a["strengths"] and a["weaknesses"] and a["sensitivity"]
        assert a["metrics"]["winner_logloss"] > 0 and 0 <= a["metrics"]["winner_top1"] <= 1
        if a["family"] == "classification":
            assert 0.5 < a["metrics"]["podium_roc_auc"] <= 1
        if a["family"] == "regression":
            assert 0 < a["metrics"]["finish_pct_mae"] < 0.5
    comp = pd.read_csv(PROJECT_ROOT / "outputs/tables/algorithm_comparison.csv")
    assert set(comp["algorithm"]) == set(SYLLABUS)


def test_each_algorithm_is_a_distinct_trained_model():
    from src.multi_sim import _algos
    bundles, _ = _algos()
    kinds = {k: type(b["model"].steps[-1][1]).__name__ if hasattr(b["model"], "steps") else type(b["model"]).__name__
             for k, b in bundles.items()}
    expected = {"logistic_regression": "LogisticRegression", "decision_tree": "DecisionTreeClassifier",
                "knn": "KNeighborsClassifier", "svm": "SVC", "naive_bayes": "GaussianNB",
                "random_forest": "RandomForestClassifier", "bagging": "BaggingClassifier",
                "adaboost": "AdaBoostClassifier", "gradient_boosting": "HistGradientBoostingClassifier",
                "neural_network": "MLPClassifier", "linear_regression": "LinearRegression", "ridge": "Ridge",
                "lasso": "Lasso", "decision_tree_regression": "DecisionTreeRegressor",
                "random_forest_regression": "RandomForestRegressor",
                "gradient_boosting_regression": "HistGradientBoostingRegressor"}
    for k, cls in expected.items():
        assert kinds[k] == cls, k


def test_every_algorithm_produces_its_own_coherent_race(all_runs):
    runs = all_runs["runs"]
    assert len(runs) == len(SYLLABUS)
    for k, run in runs.items():
        res = run["results"]
        assert sorted(r["pred_rank"] for r in res) == list(range(1, 27)), k
        assert sum(r["win_prob"] for r in res) == pytest.approx(1, abs=1e-9), k
        assert sum(r["podium_prob"] for r in res) == pytest.approx(3, abs=1e-9), k
        assert run["podium"] == [r["driver_id"] for r in res[:3]]
    probs = {k: tuple(round(r["win_prob"], 6) for r in sorted(run["results"], key=lambda x: x["driver_id"]))
             for k, run in runs.items()}
    assert len(set(probs.values())) == len(runs)                     # no two algorithms give identical predictions


def test_shared_random_numbers_and_same_scenario(all_runs):
    """All algorithms see the same scenario and the same sampled grids (common random numbers)."""
    runs = list(all_runs["runs"].values())
    ref = {r["driver_id"]: (r["est_grid"], r["pole_prob"], r["constructor_id"], r["season"]) for r in runs[0]["results"]}
    for run in runs[1:]:
        assert {r["driver_id"]: (r["est_grid"], r["pole_prob"], r["constructor_id"], r["season"])
                for r in run["results"]} == ref


def test_single_run_matches_comparison_run(client, grid, all_runs):
    for algo in ("logistic_regression", "knn", "svm"):
        s = client.post("/api/simulate-custom-race", json={"circuit_id": "monaco", "entries": grid, "algorithm": algo}).json()
        assert s["algorithm"] == algo and s["label"] == all_runs["runs"][algo]["label"]
        a = {r["driver_id"]: r["win_prob"] for r in s["results"]}
        b = {r["driver_id"]: r["win_prob"] for r in all_runs["runs"][algo]["results"]}
        assert a == b and s["results"][0]["factors"]


def test_reproducible_and_best_option(client, grid, all_runs):
    body = {"circuit_id": "monaco", "entries": grid, "algorithms": ["knn", "best"]}
    a = client.post("/api/compare-algorithms", json=body).json()
    b = client.post("/api/compare-algorithms", json=body).json()
    assert a["runs"] == b["runs"]
    assert a["algorithms"] == ["knn", all_runs["best_validated"]]


def test_agreement_summary(all_runs):
    ag = all_runs["agreement"]
    assert sum(v["votes"] for v in ag["winner_votes"]) == len(SYLLABUS)
    assert sum(v["votes"] for v in ag["podium_votes"]) == 3 * len(SYLLABUS)
    M = np.array(ag["rank_correlation"]["matrix"])
    assert M.shape == (len(SYLLABUS),) * 2 and np.allclose(np.diag(M), 1) and np.allclose(M, M.T)
    for d in ag["per_driver"]:
        assert d["min_win_prob"] <= d["mean_win_prob"] <= d["max_win_prob"] and d["best_rank"] <= d["worst_rank"]


def test_car_swap_moves_every_main_algorithm_the_right_way(client, grid):
    e = [dict(x) for x in grid]
    i = next(k for k, x in enumerate(e) if x["driver_id"] == "senna")
    j = next(k for k, x in enumerate(e) if x["driver_id"] == "hunt")
    for f in ("constructor_id", "constructor_season"):
        e[i][f], e[j][f] = e[j][f], e[i][f]
    algos = ["logistic_regression", "random_forest", "gradient_boosting", "ridge", "plackett_luce"]
    base = client.post("/api/compare-algorithms", json={"circuit_id": "monaco", "entries": grid, "algorithms": algos}).json()
    swap = client.post("/api/compare-algorithms", json={"circuit_id": "monaco", "entries": e, "algorithms": algos}).json()
    for a in algos:
        w0 = {r["driver_id"]: r["win_prob"] for r in base["runs"][a]["results"]}
        w1 = {r["driver_id"]: r["win_prob"] for r in swap["runs"][a]["results"]}
        assert w1["senna"] < w0["senna"], a                           # McLaren 1988 -> Hesketh 1974


@pytest.mark.parametrize("algo,kind", [("logistic_regression", "coefficients"), ("ridge", "coefficients"),
                                       ("decision_tree", "tree_path"), ("decision_tree_regression", "tree_path"),
                                       ("knn", "neighbours"), ("naive_bayes", "class_means"),
                                       ("plackett_luce", "coefficients"), ("svm", "global_importance"),
                                       ("random_forest", "global_importance")])
def test_model_specific_explanations(client, grid, algo, kind):
    r = client.post("/api/explain-algorithm", json={"circuit_id": "monaco", "entries": grid, "algorithm": algo,
                                                    "driver_id": "max_verstappen"})
    assert r.status_code == 200, r.text
    x = r.json()
    from src.ranking import CARNEUTRAL_GROUPS
    assert x["native"]["kind"] == kind
    assert {f["group"] for f in x["local_sensitivity"]} == set(CARNEUTRAL_GROUPS)
    if kind == "tree_path":
        assert x["native"]["steps"][-1]["leaf"] and all("threshold" in s for s in x["native"]["steps"][:-1])
    if kind == "neighbours":
        assert len(x["native"]["items"]) == 8
    if algo == "random_forest":
        assert x["impurity_importance"]


def test_algorithm_validation_errors(client, grid):
    r = client.post("/api/compare-algorithms", json={"circuit_id": "monaco", "entries": grid, "algorithms": ["magic"]})
    assert r.status_code == 422 and "Unknown algorithm" in r.json()["detail"]["message"]
    r = client.post("/api/compare-algorithms", json={"circuit_id": "monaco", "entries": grid, "algorithms": []})
    assert r.status_code == 422
    r = client.post("/api/explain-algorithm", json={"circuit_id": "monaco", "entries": grid, "algorithm": "knn",
                                                    "driver_id": "nobody"})
    assert r.status_code == 422


def test_density_threshold_flags_about_one_percent_of_real_entries():
    import joblib
    from src.driver_ratings import attach, crew_per_race, load_ratings
    from src.features import load_features
    from src.ranking import scenario_matrix
    d = joblib.load(PROJECT_ROOT / "models/algorithms/density.joblib")
    mf = crew_per_race(attach(load_features(), load_ratings(), -1))
    rows = mf[mf["season"] >= 1981]
    ll = d["model"].score_samples(d["prep"].transform(scenario_matrix(rows, d["features"])))
    assert (ll < d["threshold"]).mean() == pytest.approx(0.01, abs=0.002)


def test_car_clusters_use_silhouette_and_do_not_predict():
    cars = pd.read_csv(PROJECT_ROOT / "outputs/tables/car_clusters.csv")
    summ = pd.read_csv(PROJECT_ROOT / "outputs/tables/car_cluster_summary.csv", index_col=0)["value"]
    assert int(summ["k"]) == cars["kmeans"].nunique() == 2
    assert {"pc1", "pc2", "kmeans", "dbscan", "birch"} <= set(cars.columns)
    assert not any(c.startswith("target_") for c in cars.columns)


def test_density_warning_only_for_hypothetical_outliers(client, grid):
    """Real prime pairings are rare but real: no extrapolation warning. Elite drivers in a back-marker car are."""
    body = {"circuit_id": "monaco", "entries": grid, "algorithms": ["knn"]}
    real = client.post("/api/compare-algorithms", json=body).json()
    assert not any(w.startswith("Outside historical data") for w in real["warnings"])
    e = [dict(x) for x in grid]
    for x in e[:8]:
        x.update(constructor_id="minardi", constructor_season=1990)
    odd = client.post("/api/compare-algorithms", json={**body, "entries": e}).json()
    assert any(w.startswith("Outside historical data") for w in odd["warnings"])
