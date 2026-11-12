"""
Teammate-network driver ratings (D-021): car-neutral driver skill corrected for teammate strength.

Every pair of teammates in a race shares a car, so their gap measures driver skill only:
    gap_ij = x_i - x_j  (x = grid percentile for qualifying pace, finish percentile for race pace)
Ratings s_(driver, season) solve the regularised least-squares problem
    sum_pairs (gap_ij - (s_i - s_j))^2 + LAMBDA_FUSE * sum_d sum_t (s_(d,t) - s_(d,t-1))^2 + LAMBDA_RIDGE * sum s^2
Drivers are linked through changing teammates, so beating a strong teammate counts more than beating a
weak one, and the fusion term lets a driver's level drift smoothly across seasons.

Leakage rule: the table holds, for every as-of season F, ratings fitted on races in seasons <= F only.
Training rows of season S use as-of F = S - 1; a performance snapshot "after season S" uses F = S.
Gaps are divided by the season's RMS teammate gap (era comparability). Ratings are sign-flipped so
that HIGHER = better.

Output: data/processed/driver_network_ratings.csv (driver_id, asof_season, net_quali_skill, net_race_skill)
Run:    python -m src.driver_ratings
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr

from src.config import PROJECT_ROOT, load_config
from src.features import load_features

LAMBDA_FUSE, LAMBDA_RIDGE = 4.0, 0.05


def teammate_pairs(mf):
    d = mf[["race_id", "season", "constructor_id", "driver_id", "finish_pct"]].copy()
    d["grid_pct"] = (mf["grid_position"] - 1) / (mf["field_size"] - 1)
    rows = []
    for (_, season, _), g in d.groupby(["race_id", "season", "constructor_id"]):
        if len(g) < 2:
            continue
        for a, b in combinations(g.itertuples(index=False), 2):
            rows.append((season, a.driver_id, b.driver_id, a.grid_pct - b.grid_pct, a.finish_pct - b.finish_pct))
    out = pd.DataFrame(rows, columns=["season", "d_i", "d_j", "gap_quali", "gap_race"])
    # era comparability: express each gap in units of that season's typical (RMS) teammate gap, because
    # teammate gaps were systematically larger in early decades (bigger spreads within a team's cars and drivers)
    for c in ("gap_quali", "gap_race"):
        out[c] = out[c] / out.groupby("season")[c].transform(lambda x: np.sqrt(np.nanmean(x ** 2)))
    return out


def fit_ratings(pairs, col):
    """Ratings per (driver, season) from the given pairs; returns Series indexed by (driver, season)."""
    p = pairs[pairs[col].notna()]
    keys = pd.Index(pd.unique(pd.concat([p["d_i"] + "|" + p["season"].astype(str),
                                         p["d_j"] + "|" + p["season"].astype(str)])))
    n = len(p)
    i = keys.get_indexer(p["d_i"] + "|" + p["season"].astype(str))
    j = keys.get_indexer(p["d_j"] + "|" + p["season"].astype(str))
    r = np.arange(n)
    A = coo_matrix((np.r_[np.ones(n), -np.ones(n)], (np.r_[r, r], np.r_[i, j])), shape=(n, len(keys)))
    k = pd.DataFrame({"key": keys, "driver": keys.str.split("|").str[0],
                      "season": keys.str.split("|").str[1].astype(int)}).sort_values(["driver", "season"])
    same = k["driver"].eq(k["driver"].shift())
    a, b = keys.get_indexer(k["key"].shift()[same]), keys.get_indexer(k["key"][same])
    m = len(a)
    F = coo_matrix((np.sqrt(LAMBDA_FUSE) * np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                                                             np.r_[a, b])), shape=(m, len(keys)))
    R = coo_matrix((np.full(len(keys), np.sqrt(LAMBDA_RIDGE)), (np.arange(len(keys)), np.arange(len(keys)))))
    y = np.r_[p[col].to_numpy(), np.zeros(m + len(keys))]
    s = lsqr(vstack([A, F, R]).tocsr(), y, atol=1e-10, btol=1e-10)[0]
    return pd.Series(-s, index=pd.MultiIndex.from_arrays([k.set_index("key").loc[keys, "driver"].to_numpy(),
                                                          k.set_index("key").loc[keys, "season"].to_numpy()],
                                                         names=["driver_id", "season"]))


def asof_table(mf):
    pairs = teammate_pairs(mf)
    seasons_by_driver = mf.groupby("driver_id")["season"].apply(set)
    out = []
    for F in range(int(mf["season"].min()), int(mf["season"].max()) + 1):
        p = pairs[pairs["season"] <= F]
        if p.empty:
            continue
        rq, rr = fit_ratings(p, "gap_quali"), fit_ratings(p, "gap_race")
        for name, r in (("net_quali_skill", rq), ("net_race_skill", rr)):
            latest = r.reset_index().sort_values("season").groupby("driver_id").tail(1)   # latest season <= F
            out.append(latest.assign(asof_season=F, kind=name).rename(columns={0: "value"}))
    t = pd.concat(out).pivot_table(index=["driver_id", "asof_season"], columns="kind", values="value").reset_index()
    t.columns.name = None
    # only keep drivers who raced up to F (all, by construction) - plus a flag for how many seasons inform it
    t["rated_seasons"] = [sum(s <= F for s in seasons_by_driver.get(d, ())) for d, F in zip(t["driver_id"], t["asof_season"])]
    return t


def attach(df, ratings, season_offset):
    """Merge ratings as of (season + season_offset): -1 for training rows, 0 for end-of-season snapshots."""
    key = df["season"] + season_offset
    r = ratings.rename(columns={"asof_season": "_asof"})
    return df.assign(_asof=key).merge(r[["driver_id", "_asof", "net_quali_skill", "net_race_skill"]],
                                      on=["driver_id", "_asof"], how="left").drop(columns="_asof").set_index(df.index)


def load_ratings():
    return pd.read_csv(PROJECT_ROOT / load_config()["paths"]["data_processed"] / "driver_network_ratings.csv")


def main():
    t = asof_table(load_features())
    t.to_csv(PROJECT_ROOT / load_config()["paths"]["data_processed"] / "driver_network_ratings.csv", index=False)
    last = t[t["asof_season"] == t["asof_season"].max()].sort_values("net_quali_skill", ascending=False)
    print(f"rows {len(t)}; top quali ratings as of {t['asof_season'].max()}:")
    print(last.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()


CREW_COLS = {"net_quali_skill": "car_crew_quali_skill", "net_race_skill": "car_crew_race_skill"}


def crew_per_race(rows):
    """Training rows (ratings attached as of S-1): mean rating of the car's drivers in that race (known at T0).
    Lets the models separate how much of a team's results came from its drivers rather than its car."""
    g = rows.groupby(["race_id", "constructor_id"])
    return rows.assign(**{new: g[old].transform("mean") for old, new in CREW_COLS.items()},
                       **{new.replace("crew_", "crew_best_"): g[old].transform("max") for old, new in CREW_COLS.items()})


def crew_per_season(mf, ratings):
    """Car-season snapshots: mean rating (as of that season) of everyone who raced the car that season."""
    d = attach(mf[["season", "constructor_id", "driver_id"]].drop_duplicates(), ratings, 0)
    g = d.groupby(["constructor_id", "season"])[list(CREW_COLS)]
    best = g.max().rename(columns={k: v.replace("crew_", "crew_best_") for k, v in CREW_COLS.items()})
    return g.mean().rename(columns=CREW_COLS).join(best).reset_index()
