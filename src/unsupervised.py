"""
Unsupervised learning from the course handout, used where it is meaningful (D-022). NOT used to predict results.

1. Anomaly detection by density estimation: a Gaussian mixture model of real driver+car profiles (training rows
   1981+). A custom entry whose log-density falls below the 1% quantile of real entries is flagged as
   "outside historical data" in the simulator (every algorithm extrapolates there).
2. Clustering of car-seasons (K-Means, DBSCAN, BIRCH) on car performance profiles, shown on a 2-D PCA map:
   which cars resemble each other across eras. Clusters describe cars; they do not predict winners.

Outputs: models/algorithms/density.joblib, outputs/tables/car_clusters.csv, car_cluster_summary.csv
Run:     python -m src.unsupervised
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROJECT_ROOT, load_config
from src.driver_ratings import attach, crew_per_race, crew_per_season, load_ratings
from src.features import load_features
from src.ranking import CARNEUTRAL_GROUPS, scenario_matrix

SEED = 42
DENSITY_FEATURES = CARNEUTRAL_GROUPS["driver_skill"] + CARNEUTRAL_GROUPS["car_pace"] + CARNEUTRAL_GROUPS["car_reliability"]
DENSITY_QUANTILE = 0.01
CAR_FEATURES = ["constructor_last5_best_finish_pct", "constructor_last5_best_grid_pct", "constructor_points_share_before_race",
                "constructor_win_rate_last20", "constructor_podium_rate_last20", "constructor_dnf_rate_last10"]
K_RANGE = (2, 3)   # k chosen by silhouette; capped at 3 because the chart palette validates 3 scatter colours


def density_model(rows):
    X = scenario_matrix(rows, DENSITY_FEATURES)
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit(X)
    Z = prep.transform(X)
    fits = {n: GaussianMixture(n, covariance_type="full", random_state=SEED).fit(Z) for n in (4, 8, 12)}
    n_best = min(fits, key=lambda n: fits[n].bic(Z))
    gm = fits[n_best]
    thr = float(np.quantile(gm.score_samples(Z), DENSITY_QUANTILE))
    return {"model": gm, "prep": prep, "features": DENSITY_FEATURES, "threshold": thr, "quantile": DENSITY_QUANTILE,
            "n_components": n_best, "bic": {n: float(f.bic(Z)) for n, f in fits.items()}}


def car_clusters(mf):
    proc = PROJECT_ROOT / load_config()["paths"]["data_processed"]
    cars = pd.read_csv(proc / "constructor_snapshots.csv").merge(crew_per_season(mf, load_ratings()),
                                                                 on=["constructor_id", "season"], how="left")
    X = cars[CAR_FEATURES]
    Z = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit_transform(X)
    pca = PCA(2, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    sil = {k: float(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z))) for k in range(2, 7)}
    K = max(K_RANGE, key=lambda k: sil[k])
    km = KMeans(K, n_init=10, random_state=SEED).fit(Z)
    # DBSCAN eps from the k-distance "elbow" heuristic: 90th percentile of 5th-nearest-neighbour distances
    eps = float(np.quantile(NearestNeighbors(n_neighbors=5).fit(Z).kneighbors(Z)[0][:, -1], 0.9))
    db = DBSCAN(eps=eps, min_samples=5).fit(Z)
    bi = Birch(n_clusters=K).fit(Z)
    # name K-Means clusters by their mean points share (front-runners > midfield > back of the field)
    order = pd.Series(cars["constructor_points_share_before_race"]).groupby(km.labels_).mean().sort_values(ascending=False)
    labels = {2: ["Front-running cars", "Rest of the field"], 3: ["Front-runner cars", "Midfield cars", "Back-of-field cars"]}[K]
    names = dict(zip(order.index, labels))
    out = cars[["constructor_id", "season", "constructor_name"] + CAR_FEATURES].copy()
    out["pc1"], out["pc2"] = P[:, 0], P[:, 1]
    out["kmeans"] = [names[c] for c in km.labels_]
    out["dbscan"] = np.where(db.labels_ == -1, "Noise (unusual car)", "Cluster " + pd.Series(db.labels_).astype(str))
    bi_order = pd.Series(cars["constructor_points_share_before_race"]).groupby(bi.labels_).mean().sort_values(ascending=False)
    bnames = dict(zip(bi_order.index, [f"BIRCH group {chr(65 + j)}" + (" (strongest)" if j == 0 else "") for j in range(K)]))
    out["birch"] = [bnames[c] for c in bi.labels_]
    summary = {"silhouette_kmeans": sil, "k": K, "dbscan_eps": eps, "dbscan_clusters": int(db.labels_.max() + 1),
               "dbscan_noise_share": float((db.labels_ == -1).mean()),
               "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
               "pca_loadings": {f: [float(a), float(b)] for f, a, b in zip(CAR_FEATURES, *pca.components_)}}
    return out, summary


def main():
    cfg = load_config()
    tables = PROJECT_ROOT / cfg["paths"]["tables"]
    mf = crew_per_race(attach(load_features(), load_ratings(), -1))
    d = density_model(mf[mf["season"] >= 1981])
    joblib.dump(d, PROJECT_ROOT / cfg["paths"]["models"] / "algorithms" / "density.joblib", compress=3)
    print(time.strftime("%H:%M:%S"), f"density: GMM n={d['n_components']} threshold={d['threshold']:.2f}")
    cars, summary = car_clusters(mf)
    cars.to_csv(tables / "car_clusters.csv", index=False)
    pd.Series({k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in summary.items()}).to_csv(
        tables / "car_cluster_summary.csv", header=["value"])
    print(cars["kmeans"].value_counts().to_string(), "\n", summary)


if __name__ == "__main__":
    main()
