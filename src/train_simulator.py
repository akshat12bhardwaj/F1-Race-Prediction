"""
Simulator v2 (D-021): car-neutral race model + qualifying model + prime form + default full grids.

Protocol (pre-declared):
1. Qualifying model candidates (target = real grid order), walk-forward 2017-2022.
   Selection: highest mean per-race Spearman between predicted strength order and the real grid.
2. Car-neutral race model candidates (real grid given), walk-forward 2017-2022.
   Selection: lowest pooled winner log-loss (same rule as D-020).
3. Two-stage system (grid NOT given: sampled from the qualifying model) on 2017-2022, plus the
   previous scenario model (v1 features, real grid) for reference.
4. 2023-2025: reported as POST-HOC (these outcomes were opened on 2026-11-07, run 1 of D-020).
5. Production fit on 1981-2025; prime form per driver; grid capacity per circuit; default line-ups.

Run:  python -m src.train_simulator      (after python -m src.snapshots)
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import PROJECT_ROOT, load_config
from src.driver_ratings import attach, crew_per_race, load_ratings
from src.evaluate import race_table, summarize_races
from src.features import load_features
from src.ranking import ScenarioRanker, summarize_simulation, two_stage_simulation
from src.train_scenario import metrics, race_outputs

SEED = 42
DEV = [2017, 2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024, 2025]
START = 1981
REG_MAX_GRID = 26        # long-standing regulatory maximum number of starters (D-021)
MODERN_FROM = 1990       # grid capacity uses each circuit's largest field since this season
PRIME_MIN_SEASON_STARTS = 8
GREATS_MIN_CAREER_STARTS = 50


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def fit(cfg, train):
    return ScenarioRanker("pl_linear", half_life_seasons=cfg.get("half_life"), top_k=cfg.get("top_k"),
                          feature_set=cfg["set"], target=cfg.get("target", "target_finish_position")).fit(
        train[(train["season"] >= START) & train[cfg.get("target", "target_finish_position")].notna()])


def quali_eval(u, df):
    d = df.assign(u=u)
    rho = d.groupby("race_id").apply(lambda g: spearmanr(-g["u"], g["grid_position"], nan_policy="omit").statistic,
                                     include_groups=False)
    pole = d.groupby("race_id").apply(lambda g: -np.log(
        (np.exp(g["u"] - g["u"].max()) / np.exp(g["u"] - g["u"].max()).sum())[g["grid_position"] == 1].sum() + 1e-12)
        if (g["grid_position"] == 1).any() else np.nan, include_groups=False)
    return {"grid_spearman": rho.mean(), "pole_logloss": pole.mean()}


def two_stage_outputs(race_m, quali_m, df, n_sims=400):
    parts = []
    for i, (rid, g) in enumerate(df.groupby("race_id", sort=False)):
        s = summarize_simulation(two_stage_simulation(race_m, quali_m, g, n_sims, SEED + i))
        o = g[["race_id", "season", "driver_id", "grid_position", "target_finish_position", "target_winner",
               "target_podium", "exclude_from_winner_metrics"]].copy()
        o["win_prob"], o["podium_prob"], o["expected"] = s["win"], s["podium"], s["expected"]
        o["pred_rank"] = o.assign(neg=-o["win_prob"]).sort_values(["expected", "neg", "driver_id"]) \
            .reset_index().reset_index().set_index("index")["level_0"].reindex(o.index) + 1
        parts.append(o)
    return pd.concat(parts)


def grid_capacity(mf):
    r = mf.groupby("race_id").agg(circuit_id=("circuit_id", "first"), season=("season", "first"),
                                  starters=("driver_id", "size"))
    modern = r[r["season"] >= MODERN_FROM].groupby("circuit_id")["starters"].max()
    allt = r.groupby("circuit_id")["starters"].max()
    cap = modern.reindex(allt.index).fillna(allt).clip(upper=REG_MAX_GRID).astype(int)
    basis = np.where(allt.index.isin(modern.index), f"largest field since {MODERN_FROM}", "largest field ever (no race since 1990)")
    return pd.DataFrame({"circuit_id": allt.index, "grid_capacity": cap.values, "capacity_basis": basis})


def main():
    cfg = load_config()
    tables = PROJECT_ROOT / cfg["paths"]["tables"]
    proc = PROJECT_ROOT / cfg["paths"]["data_processed"]
    mdir = PROJECT_ROOT / cfg["paths"]["models"] / "scenario"
    ratings = load_ratings()
    mf = crew_per_race(attach(load_features(), ratings, -1))   # race in season S: ratings fitted on seasons <= S-1

    # ---------------- 1. qualifying model ----------------
    qcands = [{"name": f"{fs}_pl_hl8{'_top' + str(k) if k else ''}", "set": fs, "half_life": 8, "top_k": k,
               "target": "grid_position"} for fs in ("quali", "quali_best") for k in (None, 10)]
    qrows, qbest = [], None
    for c in qcands:
        res = [quali_eval(fit(c, mf[mf["season"] < S]).utility(mf[mf["season"] == S]), mf[mf["season"] == S])
               for S in DEV]
        row = {"model": c["name"], "grid_spearman": np.mean([r["grid_spearman"] for r in res]),
               "pole_logloss": np.mean([r["pole_logloss"] for r in res])}
        qrows.append(row)
        log(f"{c['name']:28s} grid_spearman={row['grid_spearman']:.4f} pole_ll={row['pole_logloss']:.4f}")
    qtab = pd.DataFrame(qrows)
    qsel = next(c for c in qcands if c["name"] == qtab.sort_values("grid_spearman", ascending=False).iloc[0]["model"])
    log(f"selected qualifying model: {qsel['name']}")

    # ---------------- 2. car-neutral race model (real grid) ----------------
    rcands = [{"name": f"{fs}_pl_hl8_top{k}", "set": fs, "half_life": 8, "top_k": k}
              for fs in ("carneutral", "carneutral_best") for k in (1, 3)]
    rcands.append({"name": "v1_scenario_pl_hl8_top1 (reference)", "set": "v1", "half_life": 8, "top_k": 1, "ref": True})
    rrows, outs = [], {}
    for c in rcands:
        o = pd.concat([race_outputs(mf[mf["season"] == S], fit(c, mf[mf["season"] < S]).utility(mf[mf["season"] == S]))
                       for S in DEV])
        outs[c["name"]] = o
        rrows.append({"model": c["name"], "grid": "real grid given", **metrics(o)})
        log(f"{c['name']:36s} winner_ll={rrows[-1]['winner_logloss']:.4f} rank_mae={rrows[-1]['rank_mae']:.3f}")
    rtab = pd.DataFrame(rrows)
    rsel = next(c for c in rcands if c["name"] == rtab[~rtab["model"].str.contains("reference")]
                .sort_values("winner_logloss").iloc[0]["model"])
    log(f"selected race model: {rsel['name']}")

    # ---------------- 3. two-stage system (grid predicted) ----------------
    parts = []
    for S in DEV:
        tr, te = mf[mf["season"] < S], mf[mf["season"] == S]
        parts.append(two_stage_outputs(fit(rsel, tr), fit(qsel, tr), te))
    ts = pd.concat(parts)
    rtab = pd.concat([rtab, pd.DataFrame([{"model": f"two-stage: {qsel['name']} -> {rsel['name']}",
                                           "grid": "grid predicted (simulator)", **metrics(ts)}])], ignore_index=True)
    log(f"two-stage (no grid): {rtab.iloc[-1][['winner_logloss', 'winner_top1', 'podium_overlap', 'rank_mae']].to_dict()}")

    # ---------------- 4. 2023-2025 (post-hoc) ----------------
    tr, te = mf[mf["season"] <= 2022], mf[mf["season"].isin(TEST)]
    post = [{"model": rsel["name"], "grid": "real grid given", "status": "post-hoc",
             **metrics(race_outputs(te, fit(rsel, tr).utility(te)))},
            {"model": "two-stage simulator", "grid": "grid predicted", "status": "post-hoc",
             **metrics(two_stage_outputs(fit(rsel, tr), fit(qsel, tr), te))}]
    post = pd.DataFrame(post)
    log(f"2023-2025 post-hoc:\n{post[['model', 'winner_logloss', 'winner_top1', 'podium_overlap', 'rank_mae']].round(4).to_string(index=False)}")

    # ---------------- 5. production + prime form + capacity + defaults ----------------
    race_m, quali_m = fit(rsel, mf), fit(qsel, mf)
    snaps = attach(pd.read_csv(proc / "driver_snapshots.csv", dtype={"driver_code": str}), ratings, 0)
    # prime form (D-021): the season with the highest era-normalised teammate-network rating (qualifying pace +
    # race pace, each rated using only seasons up to that one). Car-neutral by construction: it only compares
    # drivers with teammates in the same car, linked across careers.
    snaps["driver_skill_score"] = snaps["net_quali_skill"] + snaps["net_race_skill"]
    elig = snaps[(snaps["season_starts"] >= PRIME_MIN_SEASON_STARTS) & snaps["driver_skill_score"].notna()]
    prime = elig.loc[elig.groupby("driver_id")["driver_skill_score"].idxmax()]
    fallback = snaps[~snaps["driver_id"].isin(prime["driver_id"])]
    fallback = fallback.loc[fallback.groupby("driver_id")["season_starts"].idxmax()].assign(prime_low_evidence=True)
    prime = pd.concat([prime.assign(prime_low_evidence=False), fallback])
    prime = prime.rename(columns={"season": "prime_season", "constructor_id": "prime_constructor_id",
                                  "constructor_name": "prime_constructor_name"})[
        ["driver_id", "driver_first_name", "driver_last_name", "driver_code", "prime_season", "prime_constructor_id",
         "prime_constructor_name", "season_starts", "driver_career_starts", "driver_skill_score", "prime_low_evidence"]]
    prime["greats_eligible"] = (~prime["prime_low_evidence"]) & (
        prime["driver_id"].map(snaps.groupby("driver_id")["driver_career_starts"].max()) >= GREATS_MIN_CAREER_STARTS)
    prime = prime.sort_values("driver_skill_score", ascending=False).reset_index(drop=True)
    prime["prime_rank"] = np.arange(1, len(prime) + 1)
    prime.to_csv(proc / "driver_primes.csv", index=False)
    grid_capacity(mf).to_csv(proc / "circuit_grid_capacity.csv", index=False)

    joblib.dump({"race": race_m, "quali": quali_m}, mdir / "simulator_v2.joblib", compress=3)
    meta = {"version": "simulator_v2", "race_model": rsel, "quali_model": qsel,
            "race_features": race_m._cols(), "quali_features": quali_m._cols(),
            "trained_on": f"{START}-2025", "n_sims_default": 5000, "seed": SEED,
            "selection_rules": {"quali": "highest mean per-race Spearman vs real grid, walk-forward 2017-2022",
                                "race": "lowest pooled winner log-loss with real grid, walk-forward 2017-2022"},
            "prime_rule": f"season with the highest teammate-network rating (net_quali_skill + net_race_skill, "
                          f"era-normalised, fitted on seasons up to that season) among seasons with >= "
                          f"{PRIME_MIN_SEASON_STARTS} starts; otherwise the season with most starts (flagged low evidence)",
            "greats_rule": f"default line-up = drivers ranked by prime driver-skill score with >= "
                           f"{GREATS_MIN_CAREER_STARTS} career starts, up to the circuit's grid capacity",
            "grid_capacity_rule": f"largest field at the circuit since {MODERN_FROM} (or ever, if no race since), capped "
                                  f"at {REG_MAX_GRID} (long-standing regulatory maximum; 2026 rules reportedly 24, unverified)",
            "coefficients_standardised": {
                "race": dict(zip(race_m.prep_.get_feature_names_out(), race_m.pl_.coef_.round(4))),
                "quali": dict(zip(quali_m.prep_.get_feature_names_out(), quali_m.pl_.coef_.round(4)))},
            "development_2017_2022": rtab.to_dict("records"), "quali_development": qtab.to_dict("records"),
            "post_hoc_2023_2025": post.to_dict("records"), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (mdir / "metadata_v2.json").write_text(json.dumps(meta, indent=2, default=lambda x: None if x != x else float(x)))
    qtab.to_csv(tables / "simulator_quali_comparison.csv", index=False)
    rtab.to_csv(tables / "simulator_race_comparison.csv", index=False)
    post.to_csv(tables / "simulator_post_hoc_2023_2025.csv", index=False)
    log(f"top prime drivers:\n{prime.head(26)[['driver_id', 'prime_season', 'prime_constructor_name', 'driver_skill_score', 'greats_eligible']].to_string(index=False)}")


if __name__ == "__main__":
    main()
