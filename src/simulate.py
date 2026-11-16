"""
Custom race simulator v2 (Mode B; D-019, D-021).

    default_grid(circuit_id)                    the eligible 2025 field (reference_pool) in 2025 cars
    predict_custom_race(circuit_id, entries, conditions, n_sims, seed)
        entries: [{"driver_id", "season"?, "constructor_id"?, "constructor_season"?}]
        season defaults to the driver's PRIME season; the car defaults to the car of that season.

Per simulated race (seeded Monte Carlo):
  1. qualifying: grid order drawn from the qualifying Plackett-Luce model (car qualifying pace, car strength,
     teammate-network driver pace, experience, circuit history)
  2. race: finishing order drawn from the car-neutral race Plackett-Luce model, using that sampled grid,
     car race pace / reliability / circuit history and car-neutral driver skill.
Driver measures are car-neutral (teammate comparisons), so assigning any car swaps exactly the car terms.
"""

import hashlib
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.driver_ratings import attach, crew_per_season, load_ratings
from src.features import load_features
from src.ranking import GROUP_LABELS, summarize_simulation, two_stage_simulation

MAX_DRIVERS, MAX_SIMS = 26, 50_000
CROSS_ERA_YEARS, CAR_ERA_GAP = 10, 15

# ---------------- reference season (D-030) ----------------
# The data ends in 2025, so the app predicts with 2025 drivers in 2025 cars. Nothing from 2026 is used or invented.
REFERENCE_SEASON = 2025
REF_MAX_PER_TEAM = 2            # real teams enter two cars



DRIVER_COLS = ["driver_age_years", "driver_career_starts", "driver_teammate_finish_gap_last10",
               "driver_teammate_grid_gap_last10", "driver_teammate_beat_rate", "net_quali_skill", "net_race_skill"]
CAR_COLS = ["constructor_last5_avg_finish_pct", "constructor_last5_avg_grid_pct", "constructor_points_share_before_race",
            "constructor_win_rate_last20", "constructor_podium_rate_last20", "constructor_dnf_rate_last10",
            "car_crew_quali_skill", "car_crew_race_skill", "constructor_last5_best_grid_pct",
            "constructor_last5_best_finish_pct", "car_crew_best_quali_skill", "car_crew_best_race_skill"]


class ScenarioError(ValueError):
    """Invalid scenario input (message is safe to show to the user)."""


@lru_cache(maxsize=1)
def _data():
    cfg = load_config()
    proc = PROJECT_ROOT / cfg["paths"]["data_processed"]
    mdir = PROJECT_ROOT / cfg["paths"]["models"] / "scenario"
    if not (mdir / "simulator_v2.joblib").exists():
        raise FileNotFoundError("models/scenario/simulator_v2.joblib missing; run python -m src.train_simulator")
    mf = load_features()
    hist = mf[["race_id", "season", "circuit_id", "driver_id", "constructor_id", "finish_pct", "target_dnf"]].assign(
        grid_pct=(mf["grid_position"] - 1) / (mf["field_size"] - 1))
    team_race = hist.groupby(["constructor_id", "race_id"], as_index=False).agg(
        season=("season", "first"), circuit_id=("circuit_id", "first"), finish_pct=("finish_pct", "mean"),
        grid_pct=("grid_pct", "mean"))
    drivers = attach(pd.read_csv(proc / "driver_snapshots.csv", dtype={"driver_code": str}), load_ratings(), 0)
    driven = set(zip(drivers["driver_id"], drivers["constructor_id"], drivers["season"]))
    # every car a driver actually raced in a season (incl. mid-season moves) counts as a real pairing
    driven |= set(zip(mf["driver_id"], mf["constructor_id"], mf["season"]))
    teams = pd.read_csv(proc / "constructor_snapshots.csv").merge(
        crew_per_season(mf, load_ratings()), on=["constructor_id", "season"], how="left")
    circuits = pd.read_csv(proc / "circuits.csv").merge(pd.read_csv(proc / "circuit_grid_capacity.csv"), on="circuit_id")
    # circuit attrition for a NEXT race = mean DNF share over all previous races there (features.py, D-023)
    dnf = mf.groupby("race_id").agg(circuit_id=("circuit_id", "first"), s=("target_dnf", "mean")).groupby("circuit_id")["s"].mean()
    circuits["circuit_dnf_rate"] = circuits["circuit_id"].map(dnf)
    # the race number each driver actually carried that season (most used in that season's results; e.g. Verstappen
    # 1 in 2025, Senna 12 in 1988). drivers.csv only holds today's permanent number, which is wrong for past seasons.
    res = pd.read_csv(PROJECT_ROOT / cfg["paths"]["data_interim"] / "results.csv",
                      usecols=["season", "driver_id", "driver_number"]).dropna()
    numbers = res.groupby(["driver_id", "season"])["driver_number"].agg(lambda s: int(s.mode().iloc[0]))
    drivers["driver_number"] = pd.Series(list(zip(drivers["driver_id"], drivers["season"])), index=drivers.index).map(numbers)
    drivers["nationality"] = drivers["driver_id"].map(
        pd.read_csv(PROJECT_ROOT / cfg["paths"]["data_interim"] / "drivers.csv").dropna(subset=["driver_nationality"])
        .groupby("driver_id")["driver_nationality"].last())
    return {"models": joblib.load(mdir / "simulator_v2.joblib"),
            "meta": json.loads((mdir / "metadata_v2.json").read_text()),
            "drivers": drivers, "teams": teams,
            "primes": pd.read_csv(proc / "driver_primes.csv", dtype={"driver_code": str}),
            "circuits": circuits, "hist": hist, "team_race": team_race, "driven": driven,
            "latest_season": int(mf["season"].max())}


# ---------------- catalogues ----------------
def circuits():
    c = _data()["circuits"].sort_values(["last_season", "races"], ascending=False)
    return c.to_dict("records")


def available_drivers():
    D = _data()
    d = D["drivers"].sort_values("season")
    g = d.groupby("driver_id").agg(
        first_name=("driver_first_name", "last"), last_name=("driver_last_name", "last"),
        code=("driver_code", "last"), number=("driver_number", "last"), seasons=("season", list),
        teams=("constructor_id", list),   # the real car of each season, aligned with `seasons`
        career=("driver_career_starts", list), season_starts=("season_starts", list),   # for the eligibility badge
        rated=("net_race_skill", lambda s: [bool(pd.notna(v)) for v in s]),
        career_starts=("driver_career_starts", "max"))
    g["first_season"], g["last_season"] = g["seasons"].str[0], g["seasons"].str[-1]
    p = D["primes"].set_index("driver_id")
    g = g.join(p[["prime_season", "prime_constructor_id", "prime_constructor_name", "prime_rank", "prime_low_evidence"]])
    return g.reset_index().sort_values(["prime_rank", "driver_id"]).to_dict("records")


def driver_periods(driver_id):
    D = _data()
    r = D["drivers"][D["drivers"]["driver_id"] == driver_id].sort_values("season")
    if r.empty:
        raise ScenarioError(f"Unknown driver '{driver_id}'.")
    primes = D["primes"].set_index("driver_id")
    prime = int(primes.loc[driver_id, "prime_season"]) if driver_id in primes.index else None
    out = r[["season", "constructor_id", "constructor_name", "season_starts", "driver_career_starts",
             "driver_champ_pos_before_race", "driver_wins_season_before_race", "net_quali_skill", "net_race_skill"]]
    out = out.rename(columns={"driver_career_starts": "career_starts_to_date",
                              "driver_champ_pos_before_race": "championship_position",
                              "driver_wins_season_before_race": "season_wins"})
    out["is_prime"] = out["season"] == prime
    out["low_evidence"] = (out["career_starts_to_date"] < 10) | (out["season_starts"] < 5)
    return out.to_dict("records")


def constructors(season=None):
    t = _data()["teams"]
    if season is not None:
        t = t[t["season"] == season]
    return t[["constructor_id", "season", "constructor_name", "drivers", "constructor_points_share_before_race",
              "constructor_last5_avg_finish_pct", "constructor_last5_avg_grid_pct"]].sort_values(
        ["season", "constructor_points_share_before_race"], ascending=[False, False]).to_dict("records")


def default_grid(circuit_id):
    """The eligible 2025 field (see reference_pool), each driver in the car they finished 2025 with, ordered by
    2025 car strength then teammate qualifying rating. Excluded drivers are never filled in."""
    D = _data()
    if circuit_id not in set(D["circuits"]["circuit_id"]):
        raise ScenarioError(f"Unknown circuit '{circuit_id}'.")
    pool = reference_pool()
    cars = D["teams"][D["teams"]["season"] == REFERENCE_SEASON].set_index("constructor_id")
    el = sorted(pool["eligible"], key=lambda d: (-cars.loc[d["constructor_id"], "constructor_points_share_before_race"],
                                                 d["constructor_id"], -d["net_quali_skill"]))
    entries = [{"driver_id": d["driver_id"], "season": REFERENCE_SEASON, "constructor_id": d["constructor_id"],
                "constructor_season": REFERENCE_SEASON} for d in el]
    return {"circuit_id": circuit_id, "reference_season": REFERENCE_SEASON, "reference_grid_capacity": len(cars) * REF_MAX_PER_TEAM,
            "grid_capacity": capacity(circuit_id, entries),
            "historical_grid_capacity": int(D["circuits"].set_index("circuit_id").loc[circuit_id, "grid_capacity"]),
            "entries": entries, "excluded": pool["excluded"]}


# Driver features the course models read (algorithms.GROUPS["driver_skill"] minus career starts, which is never missing).
SKILL_COLS = ["net_race_skill", "net_quali_skill", "driver_teammate_finish_gap_last10", "driver_teammate_grid_gap_last10",
              "driver_teammate_beat_rate", "driver_age_years"]
MIN_CAREER, MIN_SEASON = 10, 5   # the project's existing low-evidence rule; teammate features are 10-race windows


def eligibility(s, circuit_starts):
    """How much real evidence stands behind one driver-season in a scenario (see docs/DRIVER_DATA_AUDIT.md).
    supported: >= 10 career starts, >= 5 that season and every skill feature observed (teammate comparisons).
    limited: some F1 data but below that bar; part of the driver's skill would be imputed.
    insufficient: no F1 race in the data; all skill features would be imputed."""
    career, season = int(s["driver_career_starts"]), int(s["season_starts"])
    imputed = int(sum(pd.isna(s[c]) for c in SKILL_COLS))
    level = ("insufficient" if career == 0 else
             "supported" if career >= MIN_CAREER and season >= MIN_SEASON and imputed == 0 else "limited")
    return {"eligibility": level, "evidence": {"career_starts": career, "season_starts": season,
                                               "circuit_starts": int(circuit_starts), "imputed_skill_features": imputed,
                                               "of_skill_features": len(SKILL_COLS)}}


@lru_cache(maxsize=1)
def reference_pool():
    """Drivers the app may race: every driver with a 2025 start whose 2025 driver-season is 'supported' and whose
    2025 car has a performance profile. Circuit history is NOT required (the models impute it from the training
    distribution, as for any first visit). Everyone else is listed in `excluded` with the evidence gap."""
    D = _data()
    dr = D["drivers"][D["drivers"]["season"] == REFERENCE_SEASON]
    cars = set(D["teams"].loc[D["teams"]["season"] == REFERENCE_SEASON, "constructor_id"])
    eligible, excluded = [], []
    for _, s in dr.sort_values("driver_id").iterrows():
        ev = eligibility(s, 0)["evidence"]
        gaps = ([f"{ev['career_starts']} career starts (< {MIN_CAREER})"] if ev["career_starts"] < MIN_CAREER else []) + \
               ([f"{ev['season_starts']} starts in {REFERENCE_SEASON} (< {MIN_SEASON})"] if ev["season_starts"] < MIN_SEASON else []) + \
               ([f"{ev['imputed_skill_features']} of {len(SKILL_COLS)} skill features missing"] if ev["imputed_skill_features"] else []) + \
               ([f"no {REFERENCE_SEASON} car profile for {s['constructor_id']}"] if s["constructor_id"] not in cars else [])
        row = {"driver_id": s["driver_id"], "first_name": s["driver_first_name"], "last_name": s["driver_last_name"],
               "code": s["driver_code"], "nationality": s["nationality"] if pd.notna(s["nationality"]) else None,
               "number": int(s["driver_number"]) if pd.notna(s["driver_number"]) else None,   # 2025 race number
               "constructor_id": s["constructor_id"], "constructor_name": s["constructor_name"],
               "net_quali_skill": float(s["net_quali_skill"]) if pd.notna(s["net_quali_skill"]) else None,
               "career_starts": ev["career_starts"], "season_starts": ev["season_starts"],
               "imputed_skill_features": ev["imputed_skill_features"]}
        (excluded if gaps else eligible).append(row | ({"gaps": gaps} if gaps else {}))
    return {"reference_season": REFERENCE_SEASON, "eligible": eligible, "excluded": excluded,
            "policy": {"min_career_starts": MIN_CAREER, "min_season_starts": MIN_SEASON,
                       "skill_features_required": SKILL_COLS, "car_profile_required": True,
                       "circuit_history_required": False}}


def check_reference_entries(entries):
    """The app's race: only eligible 2025 drivers, in 2025 cars, at most two per team. Raises ScenarioError."""
    ok = {d["driver_id"] for d in reference_pool()["eligible"]}
    t = _data()["teams"]
    cars = set(t.loc[t["season"] == REFERENCE_SEASON, "constructor_id"])
    bad = [e["driver_id"] for e in entries if e["driver_id"] not in ok]
    if bad:
        raise ScenarioError(f"Not in the {REFERENCE_SEASON} prediction pool (insufficient F1 evidence): {', '.join(bad)}.")
    for e in entries:
        if (e.get("season") or REFERENCE_SEASON) != REFERENCE_SEASON or \
                (e.get("constructor_season") or REFERENCE_SEASON) != REFERENCE_SEASON:
            raise ScenarioError(f"Predictions use {REFERENCE_SEASON} drivers and {REFERENCE_SEASON} cars only.")
        if e.get("constructor_id") and e["constructor_id"] not in cars:
            raise ScenarioError(f"'{e['constructor_id']}' did not race in {REFERENCE_SEASON}.")
    n = pd.Series([e["constructor_id"] for e in entries if e.get("constructor_id")], dtype=object).value_counts()
    if len(n) and n.max() > REF_MAX_PER_TEAM:
        raise ScenarioError(f"A team can enter at most {REF_MAX_PER_TEAM} cars ({n.idxmax()}: {n.max()}).")
    return [{**e, "season": REFERENCE_SEASON, "constructor_season": REFERENCE_SEASON} for e in entries]


# ---------------- historical pool (D-032): any eligible driver-season in any eligible car-season ----------------
MODEL_CAR_COLS = ["constructor_last5_avg_finish_pct", "constructor_last5_avg_grid_pct", "constructor_points_share_before_race",
                  "constructor_win_rate_last20", "constructor_podium_rate_last20", "constructor_dnf_rate_last10",
                  "car_crew_quali_skill", "car_crew_race_skill"]     # the car inputs the course models read (algorithms.FEATURES)
MIN_CAR_RACES = 5                                                   # the car's 'last 5 races' window must be real


def model_era():
    """First season of the course models' training data (models/algorithms/metadata.json)."""
    meta = json.loads((PROJECT_ROOT / load_config()["paths"]["models"] / "algorithms" / "metadata.json").read_text())
    return min(a["train_start"] for a in meta["algorithms"].values())


@lru_cache(maxsize=1)
def history_pool():
    """One pool for every era (D-032). Rules, the same for every driver and car:
    driver-season: season inside the models' training era (1981-2025), >= 10 career starts, >= 5 starts that season and
      all six driver-skill features observed (teammate comparisons: no imputed ability) -> eligibility() 'supported'.
      Each driver races in their 'form season': their prime season if eligible, else their best eligible season.
    car-season: season inside the training era, >= 5 races that season and every model car feature observed.
    Driver skill is car-neutral (teammate-network rating on one connected scale); car features are rates and percentiles
    relative to that car's own season field. Nothing is imputed and no raw lap times are compared across eras."""
    D, start = _data(), model_era()
    d, t = D["drivers"], D["teams"]
    races = D["team_race"].groupby(["constructor_id", "season"]).size()
    ok_car = t[(t["season"] >= start) & t[MODEL_CAR_COLS].notna().all(axis=1)
               & (pd.Series(list(zip(t["constructor_id"], t["season"])), index=t.index).map(races).fillna(0) >= MIN_CAR_RACES)]
    cars = ok_car.sort_values(["season", "constructor_points_share_before_race"], ascending=[False, False])
    car_keys = set(zip(cars["constructor_id"], cars["season"]))
    sup = d[(d["season"] >= start) & (d.apply(lambda s: eligibility(s, 0)["eligibility"], axis=1) == "supported")]
    primes = D["primes"].set_index("driver_id")["prime_season"]
    drivers, excluded = [], {"before_model_era": 0, "thin_evidence": 0}
    for did, g in sup.groupby("driver_id"):
        prime = primes.get(did)
        s = g[g["season"] == prime].iloc[0] if prime in set(g["season"]) else g.loc[g["net_race_skill"].idxmax()]
        drivers.append({"driver_id": did, "first_name": s["driver_first_name"], "last_name": s["driver_last_name"],
                        "code": s["driver_code"] if pd.notna(s["driver_code"]) else None,
                        "nationality": s["nationality"] if pd.notna(s["nationality"]) else None,
                        "season": int(s["season"]), "seasons": sorted(int(x) for x in g["season"]),
                        "number": int(s["driver_number"]) if pd.notna(s["driver_number"]) else None,
                        "constructor_id": s["constructor_id"], "constructor_name": s["constructor_name"],
                        "own_car_eligible": (s["constructor_id"], int(s["season"])) in car_keys,
                        "real_cars": sorted([c, int(y)] for (dd, c, y) in D["driven"] if dd == did and int(y) in set(g["season"])),
                        "net_race_skill": float(s["net_race_skill"])})
    ids = {x["driver_id"] for x in drivers}
    for did, g in d.groupby("driver_id"):
        if did not in ids:
            excluded["before_model_era" if g["season"].max() < start else "thin_evidence"] += 1
    drivers.sort(key=lambda x: (-x["season"], x["last_name"]))
    return {"model_era": [start, int(d["season"].max())], "drivers": drivers,
            "cars": [{"constructor_id": r["constructor_id"], "season": int(r["season"]), "constructor_name": r["constructor_name"],
                      "drivers": r["drivers"], "points_share": float(r["constructor_points_share_before_race"])}
                     for _, r in cars.iterrows()],
            "excluded": excluded,
            "policy": {"model_era_start": start, "min_career_starts": MIN_CAREER, "min_season_starts": MIN_SEASON,
                       "skill_features_required": SKILL_COLS, "car_features_required": MODEL_CAR_COLS,
                       "min_car_races": MIN_CAR_RACES, "max_cars_per_car_season": REF_MAX_PER_TEAM}}


def check_entries(entries):
    """Validate an app grid against history_pool(); fills each entry's default season and car. Raises ScenarioError
    with a short reason (the combination is rejected, never patched with invented data)."""
    P = history_pool()
    by = {x["driver_id"]: x for x in P["drivers"]}
    cars = {(c["constructor_id"], c["season"]) for c in P["cars"]}
    lo = P["model_era"][0]
    out = []
    for e in entries:
        d = by.get(e["driver_id"])
        if d is None:
            raise ScenarioError(f"{e['driver_id']} is not in the prediction pool (needs a {lo}-2025 season with full "
                                "teammate-comparison data).")
        season = int(e.get("season") or d["season"])
        if season not in d["seasons"]:
            raise ScenarioError(f"{d['first_name']} {d['last_name']}: no eligible {season} season "
                                f"(eligible: {', '.join(map(str, d['seasons']))}).")
        own = _team_of(e["driver_id"], season)
        cid = e.get("constructor_id") or own
        cs = int(e.get("constructor_season") or season)
        if (cid, cs) not in cars:
            raise ScenarioError(f"No eligible {cs} car for '{cid}' (needs {MIN_CAR_RACES}+ races that season "
                                f"within {lo}-2025 and complete car data).")
        out.append({**e, "season": season, "constructor_id": cid, "constructor_season": cs})
    n = pd.Series([f"{e['constructor_id']} {e['constructor_season']}" for e in out], dtype=object).value_counts()
    if len(n) and n.max() > REF_MAX_PER_TEAM:
        raise ScenarioError(f"A car can be entered at most {REF_MAX_PER_TEAM} times ({n.idxmax()}: {n.max()}).")
    return out


# ---------------- the app's race (D-036): 22 drivers, 11 constructors, two each, every constructor in its peak car ----------------
GRID_SIZE, GRID_TEAMS, SEATS = 22, 11, 2
PEAK_WEIGHTS = {"grid_pct": 0.4, "finish_pct": 0.4, "dnf_vs_field": 0.2}   # lower is better for all three
PEAK_MIN_RACES = 8              # a peak needs most of a season behind it; fewer -> only if the team has nothing else


@lru_cache(maxsize=1)
def peak_cars():
    """One coherent car-season per constructor: its strongest eligible car-season (history_pool cars) by
      peak_score = -(0.4 grid_pct + 0.4 finish_pct + 0.2 (DNF share - that season's field DNF share)),
    season means over all the car's starts. grid_pct / finish_pct are within-race percentiles (0 = pole / win, 1 = last
    of that race's field), so every era is measured against its own contemporaries; reliability is taken relative to
    the season's field because DNF rates fell from ~40% (1980s) to ~10% (2020s). No raw lap times, no win counts.
    (Re-standardising the percentiles per season was rejected: it rewards seasons with many backmarker teams.)
    The selected season's real snapshot features are used unchanged;
    features from different seasons are never mixed. Uses results 1981-2025 only (the app's hypothetical races);
    model training and evaluation are untouched, so no evaluation window sees it."""
    D, P = _data(), history_pool()
    s = D["hist"].groupby(["season", "constructor_id"]).agg(
        grid_pct=("grid_pct", "mean"), finish_pct=("finish_pct", "mean"), dnf_rate=("target_dnf", "mean"),
        races=("race_id", "nunique")).reset_index()
    s["dnf_vs_field"] = s["dnf_rate"] - s["season"].map(D["hist"].groupby("season")["target_dnf"].mean())
    s["peak_score"] = -sum(w * s[c] for c, w in PEAK_WEIGHTS.items())
    elig = {(c["constructor_id"], c["season"]) for c in P["cars"]}
    names = {(c["constructor_id"], c["season"]): c["constructor_name"] for c in P["cars"]}
    s = s[[k in elig for k in zip(s["constructor_id"], s["season"])]]
    out = []
    for cid, g in s.groupby("constructor_id"):
        full = g[g["races"] >= PEAK_MIN_RACES]
        cand = (full if len(full) else g).sort_values(["peak_score", "season"], ascending=[False, False])
        b = cand.iloc[0]
        out.append({"constructor_id": cid, "constructor_name": names[(cid, int(b["season"]))], "season": int(b["season"]),
                    "peak_score": round(float(b["peak_score"]), 4),
                    **{c: round(float(b[c]), 4) for c in [*PEAK_WEIGHTS, "dnf_rate"]}, "races": int(b["races"]),
                    "seasons_considered": len(g), "low_evidence": not len(full),
                    "runner_up": None if len(cand) < 2 else {"season": int(cand.iloc[1]["season"]),
                                                             "peak_score": round(float(cand.iloc[1]["peak_score"]), 4)}})
    return sorted(out, key=lambda c: c["constructor_name"].casefold())


@lru_cache(maxsize=1)
def peak_default_grid():
    """Default 22-car grid. Rule: the 11 constructors with the highest peak_score, each with the two pool drivers who
    made the most starts in that very car-season (a pairing that really raced that car); a car whose real drivers are
    already used or not in the pool is skipped for the next-best one. Grid order: car peak_score, then driver race
    rating. Each driver still races in their own form season, so this is a hypothetical all-time field, not a real
    championship line-up."""
    D, P = _data(), history_pool()
    pool = {d["driver_id"]: d for d in P["drivers"]}
    starts = D["hist"].groupby(["constructor_id", "season", "driver_id"]).size()
    used, teams = set(), []
    for car in sorted(peak_cars(), key=lambda c: -c["peak_score"]):
        st = starts.loc[(car["constructor_id"], car["season"])].sort_values(ascending=False)
        two = [d for d in st.index if d in pool and d not in used][:SEATS]
        if len(two) == SEATS:
            teams.append((car, sorted(two, key=lambda d: -pool[d]["net_race_skill"])))
            used |= set(two)
        if len(teams) == GRID_TEAMS:
            break
    return [{"driver_id": d, "season": pool[d]["season"], "constructor_id": car["constructor_id"],
             "constructor_season": car["season"]} for car, two in teams for d in two]


def check_app_grid(entries):
    """The app's race: exactly 22 different pool drivers, 11 constructors with exactly two cars each. The user picks only
    drivers and constructors; each driver races in their form season and each constructor in its peak car-season
    (filled in here; any other season is rejected, not silently replaced). Raises ScenarioError."""
    P = history_pool()
    by, peaks = {d["driver_id"]: d for d in P["drivers"]}, {c["constructor_id"]: c for c in peak_cars()}
    if len(entries) != GRID_SIZE:
        raise ScenarioError(f"A race needs exactly {GRID_SIZE} drivers ({len(entries)} on the grid).")
    ids = [e.get("driver_id") for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ScenarioError(f"Duplicate driver(s): {', '.join(dup)}. Each driver can appear once.")
    out = []
    for e in entries:
        d, cid = by.get(e["driver_id"]), e.get("constructor_id")
        if d is None:
            raise ScenarioError(f"{e['driver_id']} is not in the prediction pool.")
        if cid not in peaks:
            raise ScenarioError(f"{d['first_name']} {d['last_name']}: choose one of the eligible constructors.")
        if e.get("season") not in (None, d["season"]):
            raise ScenarioError(f"{d['first_name']} {d['last_name']} races in their form season {d['season']}.")
        if e.get("constructor_season") not in (None, peaks[cid]["season"]):
            raise ScenarioError(f"{peaks[cid]['constructor_name']} races its peak {peaks[cid]['season']} car.")
        out.append({**e, "season": d["season"], "constructor_id": cid, "constructor_season": peaks[cid]["season"]})
    n = pd.Series([e["constructor_id"] for e in out], dtype=object).value_counts()
    if len(n) != GRID_TEAMS or (n != SEATS).any():
        odd = ", ".join(f"{peaks[c]['constructor_name']}: {k}" for c, k in n.items() if k != SEATS)
        raise ScenarioError(f"A race needs {GRID_TEAMS} constructors with {SEATS} drivers each "
                            f"({len(n)} constructors{'; ' + odd if odd else ''}).")
    return check_entries(out)


def _team_of(driver_id, season):
    d = _data()["drivers"]
    return d.loc[(d["driver_id"] == driver_id) & (d["season"] == season), "constructor_id"].iloc[0]


def capacity(circuit_id, entries):
    """Cars allowed: the largest field the circuit has held, and never fewer than the app's fixed 22-car format (D-036;
    newer circuits only ever held 20 in the data, and the grid layout is generated by the app)."""
    return max(int(_data()["circuits"].set_index("circuit_id").loc[circuit_id, "grid_capacity"]), GRID_SIZE)


# ---------------- race conditions (D-033) ----------------
# User conditions = the pre-start weather the models were trained on (3 h to 1 h before the start, reanalysis).
# Ranges: 1st-99th percentile of 807 races 1981-2025. Rain is mm per hour; the models use the 2-hour sum.
COND_RANGES = {"rain_mm_h": (0.0, 4.0), "air_temp_c": (10.0, 33.0), "humidity_pct": (20.0, 95.0), "wind_kmh": (0.0, 35.0)}
WET_MM_H = 0.25                 # = the 0.5 mm per 2 h wet threshold of the training data (config weather.wet_precip_threshold_mm)
# controls offered in the app (set after validation, D-033): each must be an input of at least one installed model
COND_CONTROLS = ["rain_mm_h", "air_temp_c", "humidity_pct", "wind_kmh"]
MIN_HUMIDITY_RAIN = 45.0        # no race with rain before the start had humidity below 48.7%
COND_PRESETS = {                # medians of the training races in each state
    "dry": {"rain_mm_h": 0.0, "air_temp_c": 21.1, "humidity_pct": 60.7, "wind_kmh": 10.8},
    "damp": {"rain_mm_h": 0.1, "air_temp_c": 19.5, "humidity_pct": 75.0, "wind_kmh": 10.8},
    "wet": {"rain_mm_h": 0.65, "air_temp_c": 18.8, "humidity_pct": 81.3, "wind_kmh": 10.8},
}


def track_state(rain_mm_h):
    return "dry" if rain_mm_h <= 0 else "wet" if rain_mm_h >= WET_MM_H else "damp"


def conditions_of(c):
    """Validated race conditions from a request: a preset ('weather': dry/damp/wet) overridden by explicit values."""
    c = dict(c or {})
    w = c.get("weather") or "dry"
    if w not in COND_PRESETS:
        raise ScenarioError("weather must be 'dry', 'damp' or 'wet'.")
    out = dict(COND_PRESETS[w])
    for k, (lo, hi) in COND_RANGES.items():
        if c.get(k) is not None:
            v = float(c[k])
            if not (lo <= v <= hi) or v != v:
                raise ScenarioError(f"{k} must be between {lo:g} and {hi:g} (range seen in 1981-2025 races).")
            out[k] = round(v, 2)
    if out["rain_mm_h"] > 0 and out["humidity_pct"] < MIN_HUMIDITY_RAIN:
        raise ScenarioError(f"Rain with humidity below {MIN_HUMIDITY_RAIN:g}% never occurred in the data; raise humidity or set rain to 0.")
    return out | {"weather": track_state(out["rain_mm_h"])}


# ---------------- scenario construction ----------------
def build_scenario(circuit_id, entries, conditions=None, n_sims=5000):
    """Scenario rows (one per entry) + warnings. Raises ScenarioError on invalid input."""
    D = _data()
    if circuit_id not in set(D["circuits"]["circuit_id"]):
        raise ScenarioError(f"Unknown circuit '{circuit_id}'.")
    if not isinstance(entries, list) or not 2 <= len(entries) <= MAX_DRIVERS:
        raise ScenarioError(f"A race needs between 2 and {MAX_DRIVERS} drivers.")
    ids = [e.get("driver_id") for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ScenarioError(f"Duplicate driver(s): {', '.join(dup)}. Each driver can appear once.")
    weather = conditions_of(conditions)               # dict: state + the four model inputs
    if not 100 <= int(n_sims) <= MAX_SIMS:
        raise ScenarioError(f"n_sims must be between 100 and {MAX_SIMS}.")
    circ = D["circuits"].set_index("circuit_id").loc[circuit_id]
    cap = capacity(circuit_id, entries)
    if len(entries) > cap:
        raise ScenarioError(f"{circ['circuit_name']} takes at most {cap} cars in this scenario "
                            "(effective simulator capacity).")
    drv, teams, hist, team_race = D["drivers"], D["teams"], D["hist"], D["team_race"]
    primes = D["primes"].set_index("driver_id")
    rows, warnings, never, era_gap, no_history = [], [], [], [], []
    for e in entries:
        did = e["driver_id"]
        snap = drv[drv["driver_id"] == did]
        if snap.empty:
            raise ScenarioError(f"Unknown driver '{did}'.")
        season = e.get("season") or (int(primes.loc[did, "prime_season"]) if did in primes.index
                                     else int(snap["season"].max()))
        s = snap[snap["season"] == season]
        if s.empty:
            raise ScenarioError(f"No {season} data for {did}: they did not start a race that season. "
                                f"Available: {', '.join(map(str, snap['season'].tolist()))}.")
        s = s.iloc[0]
        tid = e.get("constructor_id") or s["constructor_id"]
        tseason = int(e.get("constructor_season") or season)
        t = teams[(teams["constructor_id"] == tid) & (teams["season"] == tseason)]
        if t.empty:
            raise ScenarioError(f"No car data for constructor '{tid}' in {tseason}.")
        t = t.iloc[0]
        name = f"{s['driver_first_name']} {s['driver_last_name']}"
        dh = hist[(hist["driver_id"] == did) & (hist["circuit_id"] == circuit_id) & (hist["season"] <= season)]
        th = team_race[(team_race["constructor_id"] == tid)
                       & (team_race["circuit_id"] == circuit_id) & (team_race["season"] <= tseason)]
        row = {"driver_id": did, "driver_name": name, "driver_code": s["driver_code"], "season": int(season),
               "driver_number": None if pd.isna(s["driver_number"]) else int(s["driver_number"]),
               "form_basis": None if pd.isna(s.get("form_basis")) else int(s["form_basis"]),
               "car_basis": None if pd.isna(t.get("car_basis")) else t["car_basis"],
               "is_prime": did in primes.index and int(season) == int(primes.loc[did, "prime_season"]),
               "constructor_id": tid, "constructor_name": t["constructor_name"], "constructor_season": tseason,
               "real_pairing": (did, tid, tseason) in D["driven"],
               "circuit_pole_win_rate": float(circ["circuit_pole_win_rate"]),
               "circuit_dnf_rate": float(circ["circuit_dnf_rate"]),
               "wx_prestart_wet": 1.0 if 2 * weather["rain_mm_h"] >= 2 * WET_MM_H else 0.0,
               "wx_prestart_precip_sum_mm": 2 * weather["rain_mm_h"], "wx_prestart_temp_mean_c": weather["air_temp_c"],
               "wx_prestart_humidity_mean_pct": weather["humidity_pct"], "wx_prestart_wind_speed_mean_kmh": weather["wind_kmh"],
               "driver_circuit_starts": len(dh), "constructor_circuit_avg_finish_pct": th["finish_pct"].mean(),
               "constructor_circuit_avg_grid_pct": th["grid_pct"].mean(),
               **{c: s[c] for c in DRIVER_COLS}, **{c: t[c] for c in CAR_COLS}}
        row |= eligibility(s, len(dh))
        rows.append(row)
        if s["driver_career_starts"] < 10 or s["season_starts"] < 5:
            warnings.append(f"{name} ({season}): low evidence ({int(s['driver_career_starts'])} career starts, "
                            f"{int(s['season_starts'])} that season); estimate is uncertain.")
        if pd.isna(s["net_quali_skill"]):
            warnings.append(f"{name} ({season}) has no teammate comparison yet, so driver skill is imputed.")
        if not row["real_pairing"]:
            never.append(name)
        if abs(tseason - season) > CAR_ERA_GAP:
            era_gap.append(f"{name} ({season}) in a {tseason} car")
        if len(dh) == 0:
            no_history.append(name)
    df = pd.DataFrame(rows)
    if never:
        warnings.insert(0, f"{len(never)} of {len(df)} driver–car pairings never raced together in F1 "
                           f"({', '.join(never[:6])}{'…' if len(never) > 6 else ''}). Driver skill comes only from "
                           "comparisons with teammates in the same car, and car strength from that car's season, "
                           "assuming the two add up with no driver–car chemistry.")
    if era_gap:
        warnings.append("Large driver/car era gaps: " + "; ".join(era_gap[:6]) + (" …" if len(era_gap) > 6 else "")
                        + ". Car strength is relative to its own season's field, not an absolute lap time.")
    shared = df.groupby(["constructor_id", "constructor_season"]).size()
    over = shared[shared > 2]
    if len(over):
        warnings.append("More than two drivers share the same car-season (" + ", ".join(
            f"{df[(df['constructor_id'] == c) & (df['constructor_season'] == y)]['constructor_name'].iloc[0]} {y}: {n}"
            for (c, y), n in over.items()) + "); real teams entered at most two cars.")
    if no_history:
        warnings.append(f"No previous race at this circuit for {len(no_history)} driver(s) "
                        f"({', '.join(no_history[:6])}{'…' if len(no_history) > 6 else ''}); zero prior starts is recorded. Missing constructor track history is handled by each model.")
    yrs = list(df["season"]) + list(df["constructor_season"])
    if max(yrs) - min(yrs) > CROSS_ERA_YEARS:
        warnings.append(f"Cross-era field ({min(yrs)}–{max(yrs)}): drivers and cars are measured relative to their own "
                        "era. Such races never happened, so these probabilities cannot be checked against real results.")
    if circ["last_season"] < 2000:
        warnings.append(f"{circ['circuit_name']} last hosted F1 in {int(circ['last_season'])}; circuit statistics are historical.")
    return df, warnings, weather


def predict_custom_race(circuit_id, entries, conditions=None, n_sims=5000, seed=42, algorithm=None,
                        grid_mode="qualifying"):
    """algorithm given (or a fixed grid) -> that course algorithm (src/multi_sim.py, D-022); None -> the original
    two-stage Plackett-Luce path below (kept for reference and tests).
    grid_mode: 'qualifying' samples the grid from the qualifying model; 'fixed' uses the entry order as the grid."""
    if algorithm is not None or grid_mode == "fixed":
        from src.multi_sim import predict_single
        return predict_single(circuit_id, entries, algorithm or "plackett_luce", conditions, n_sims, seed, grid_mode)
    D = _data()
    df, warnings, weather = build_scenario(circuit_id, entries, conditions, n_sims)
    race_m, quali_m = D["models"]["race"], D["models"]["quali"]
    sim = two_stage_simulation(race_m, quali_m, df, int(n_sims), int(seed))
    p = summarize_simulation(sim)
    n = len(df)
    df["win_prob"], df["podium_prob"], df["top5_prob"] = p["win"], p["podium"], p["top5"]
    df["expected_position"], df["pole_prob"], df["expected_grid"] = p["expected"], p["pole"], p["expected_grid"]
    # most likely qualifying order (decreasing qualifying strength) = the displayed estimated grid
    df["est_grid"] = pd.Series(-sim["u_quali"]).rank(method="first").astype(int).to_numpy()
    df["quali_strength"] = sim["u_quali"]
    # predicted finishing order: by expected position, ties -> higher win probability -> driver id (unique)
    df["pred_rank"] = df.assign(nw=-df["win_prob"]).sort_values(["expected_position", "nw", "driver_id"]) \
        .reset_index().reset_index().set_index("index")["level_0"].reindex(df.index) + 1
    df["prob_predicted_position"] = [p["dist"][i, int(r) - 1] for i, r in enumerate(df["pred_rank"])]
    contrib = race_m.contributions(df.assign(grid_position=df["est_grid"].astype(float)))
    se = lambda q: float(np.sqrt(q * (1 - q) / n_sims))
    results = []
    for i, r in df.iterrows():
        factors = sorted(({"group": g, "label": GROUP_LABELS[g], "effect": round(float(contrib.loc[i, g]), 3)}
                          for g in contrib), key=lambda f: -abs(f["effect"]))
        results.append({
            **{k: r[k] for k in ["driver_id", "driver_name", "driver_code", "season", "constructor_id",
                                 "constructor_name", "constructor_season"]},
            "is_prime": bool(r["is_prime"]), "real_pairing": bool(r["real_pairing"]),
            "est_grid": int(r["est_grid"]), "expected_grid": float(r["expected_grid"]), "pole_prob": float(r["pole_prob"]),
            "pred_rank": int(r["pred_rank"]), "win_prob": float(r["win_prob"]), "podium_prob": float(r["podium_prob"]),
            "top5_prob": float(r["top5_prob"]), "expected_position": float(r["expected_position"]),
            "prob_predicted_position": float(r["prob_predicted_position"]),
            "win_prob_se": se(r["win_prob"]), "podium_prob_se": se(r["podium_prob"]),
            "position_distribution": [float(x) for x in p["dist"][i]],
            "factors": factors,
            "inputs": {k: (None if pd.isna(r[k]) else float(r[k])) for k in
                       ["net_quali_skill", "net_race_skill", "driver_teammate_beat_rate", "driver_career_starts",
                        "constructor_last5_avg_grid_pct", "constructor_last5_avg_finish_pct",
                        "constructor_points_share_before_race", "constructor_dnf_rate_last10", "driver_circuit_starts"]},
        })
    results.sort(key=lambda x: x["pred_rank"])
    circ = D["circuits"].set_index("circuit_id").loc[circuit_id]
    canon = json.dumps({"c": circuit_id, "e": sorted([[r["driver_id"], r["season"], r["constructor_id"],
                                                       r["constructor_season"]] for r in results]),
                        "w": weather, "n": int(n_sims), "s": int(seed)}, sort_keys=True)
    meta = D["meta"]
    return {
        "scenario_id": hashlib.sha1(canon.encode()).hexdigest()[:12],
        "circuit": {"circuit_id": circuit_id, "circuit_name": circ["circuit_name"], "country": circ["country"],
                    "race_name": circ["race_name"], "pole_win_rate": float(circ["circuit_pole_win_rate"]),
                    "races": int(circ["races"]), "historical_grid_capacity": int(circ["grid_capacity"]),
                    "grid_capacity": capacity(circuit_id, entries)},
        "conditions": weather, "n_sims": int(n_sims), "seed": int(seed), "n_drivers": n,
        "results": results,
        "podium": [r["driver_id"] for r in results[:min(3, n)]],
        "checks": {"win_prob_sum": float(df["win_prob"].sum()), "podium_prob_sum": float(df["podium_prob"].sum()),
                   "expected_podium_slots": min(3, n)},
        "warnings": warnings,
        "methodology": {
            "model": f"{meta['quali_model']['name']} → {meta['race_model']['name']}", "trained_on": meta["trained_on"],
            "summary": "Two-stage Plackett–Luce simulation. Each simulated race first draws a qualifying order from "
                       "the qualifying model (car qualifying pace and strength, driver pace vs. teammates, experience, "
                       "circuit history), then draws the finishing order from the race model using that grid "
                       "(grid slot, car race pace, reliability, circuit history, car-neutral driver skill, weather). "
                       "All probabilities are frequencies over the simulated races.",
            "prime_rule": meta["prime_rule"], "grid_capacity_rule": meta["grid_capacity_rule"],
            "assumptions": [
                "Driver skill is car-neutral: it is measured only against teammates in the same car (teammate-network "
                "rating), so drivers with unusually strong teammates may be underrated.",
                "Driver and car effects add on the log-strength scale; there is no driver–car chemistry.",
                "Car strength is relative to its own season's field (finish/grid percentiles, points share), "
                "not an absolute lap time, so cross-era car comparisons are extrapolations.",
                "Independence of irrelevant alternatives: the relative odds of two drivers do not depend on who else races.",
                "Retirements are part of the finishing-order randomness; weather acts through how strongly the grid holds.",
            ],
        },
    }
