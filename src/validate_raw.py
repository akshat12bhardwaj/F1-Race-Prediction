"""
Step 7: complete validation of the collected data foundation (READ-ONLY).

Validates races, results, qualifying, sprints, drivers, constructors, the
approved policy columns (P-01 ... P-11) and the weather data (incl. P-12),
checks cross-table consistency and classifies every missing value.

It never downloads anything (no API client is imported), never changes an
input file, and never imputes, recodes, deduplicates or removes records.

Run from the project root:   python -m src.validate_raw

Outputs (outputs/tables/):
    raw_validation_summary.csv     every check: expected vs observed, PASS / FAIL / INFO
    raw_missingness.csv            every column with missing values, split into 4 classes
    raw_cross_table_validation.csv key relationships between tables (orphans both ways)
    raw_anomalies.csv              records that are unusual (known & handled, or needing review)
"""

import hashlib

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.policies import annotate_results, is_dnf, result_category_detail
from src.weather import PLAUSIBLE, align, race_start, window_summary

TABLES = ["races", "results", "qualifying", "sprints", "drivers", "constructors",
          "results_policy", "races_policy", "weather_race", "weather_hourly"]
RACE = ["season", "round"]
DRIVER_RACE = ["season", "round", "driver_id"]
FINISHED_STATUS = r"^(Finished|Lapped|\+\d+ Laps?)$"
LAP_TIME = r"^(\d+:)?\d{1,2}\.\d{3}$"          # "1:29.179" or "53.904"

# Documented values (docs/DECISIONS_LOG.md: F-003 ... F-006, P-01 ... P-12)
EXPECTED = {
    "races": 1149, "indianapolis_500_races": 11, "eligible_races": 1138, "us_gp_at_indianapolis": 8,
    "result_rows": 25873, "multi_entry_rows": 172, "multi_entry_groups": 83, "secondary_entries": 89,
    "shared_position_rows": 228, "shared_position_races": 46, "shared_win_races": 3,
    "grid_zero_rows": 267, "pit_lane_starts": 48, "grid_unknown": 17,
    "finished": 15554, "classified_not_running": 678, "started_not_classified": 9141, "disqualified": 162,
    "did_not_start": 338, "ambiguous": 24, "dnf_rows": 9141, "status_values": 136,
    "model_population_rows": 25066, "order_exception_rows": 16, "order_exception_races": 8,
    "qualifying_rows": 10973, "races_without_qualifying": 631, "sprint_rows": 480, "sprint_races": 24,
    "unique_drivers": 817, "unique_constructors": 203, "constructor_ids_reused_after_gap": 20,
    "weather_ok": 1138, "weather_excluded": 11, "weather_failed": 0, "weather_model_use": 1138,
    "start_official_utc": 381, "start_p12_local_clock": 37, "start_imputed": 731, "weather_hourly_rows": 14794,
}
ORDER_EXCEPTION_SEASONS = [1951, 1968, 1971, 1981, 1983, 1989, 1991, 1992]


class Report:
    """Collects check results and anomalies."""

    def __init__(self):
        self.checks, self.anomalies = [], []

    def check(self, area, check, observed, expected=None, policy="", detail=""):
        if expected is None:
            status = "INFO"
        else:
            status = "PASS" if _same(observed, expected) else "FAIL"
        self.checks.append({"area": area, "check": check, "expected": _fmt(expected), "observed": _fmt(observed),
                            "status": status, "policy": policy, "detail": detail})
        return status == "PASS"

    def anomaly(self, area, anomaly, severity, policy, rows=None, detail="", **fields):
        """severity: 'known_handled' (documented + covered by a policy), 'review', or 'error'."""
        base = {"area": area, "anomaly": anomaly, "severity": severity, "policy": policy}
        if rows is not None and len(rows):
            for _, r in rows.iterrows():
                self.anomalies.append({**base, "season": r.get("season"), "round": r.get("round"),
                                       "race_id": r.get("race_id"), "driver_id": r.get("driver_id"),
                                       "detail": detail(r) if callable(detail) else detail})
        elif rows is None:
            self.anomalies.append({**base, "season": fields.get("season"), "round": fields.get("round"),
                                   "race_id": fields.get("race_id"), "driver_id": fields.get("driver_id"),
                                   "detail": detail})


def _same(a, b):
    if isinstance(b, (list, tuple, set)):
        return list(a) == list(b)
    return a == b


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (list, tuple, set)):
        return ", ".join(map(str, v))
    return str(v)


def load(interim_dir):
    return {n: pd.read_csv(interim_dir / f"{n}.csv", dtype=str, keep_default_na=False, na_values=[""]) for n in TABLES}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def flag(s):
    return s == "True"


# ----------------------------------------------------------------------
# 1. Races
# ----------------------------------------------------------------------
def validate_races(t, rep):
    ra, rp = t["races"], t["races_policy"]
    a = "races"
    rep.check(a, "race count", len(ra), EXPECTED["races"])
    rep.check(a, "race_id unique", bool(ra["race_id"].is_unique), True)
    rep.check(a, "(season, round) unique", bool(~ra.duplicated(RACE).any()), True)
    rep.check(a, "race_id == season_round(2 digits)",
              int((ra["race_id"] != ra["season"] + "_" + ra["round"].str.zfill(2)).sum()), 0)
    for col in ["race_id", "season", "round", "race_name", "race_date", "circuit_id", "circuit_lat", "circuit_lon"]:
        rep.check(a, f"missing {col}", int(ra[col].isna().sum()), 0)
    dates = pd.to_datetime(ra["race_date"], errors="coerce", format="%Y-%m-%d")
    rep.check(a, "unparsable race_date", int(dates.isna().sum()), 0)
    rep.check(a, "race_date year == season", int((dates.dt.year != num(ra["season"])).sum()), 0)
    seasons = num(ra["season"])
    rep.check(a, "seasons covered", f"{seasons.min()}-{seasons.max()} ({seasons.nunique()})", "1950-2025 (76)")
    rounds = ra.assign(r=num(ra["round"])).groupby("season")["r"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    rep.check(a, "rounds are 1..N in every season", int((~rounds).sum()), 0, detail="seasons with gaps or duplicates")
    ordered = ra.assign(d=dates, r=num(ra["round"])).sort_values(["season", "r"]).groupby("season")["d"] \
        .apply(lambda d: bool((d.diff().dropna().dt.days > 0).all()))
    rep.check(a, "dates strictly increase with round in every season", int((~ordered).sum()), 0)
    lat, lon = num(ra["circuit_lat"]), num(ra["circuit_lon"])
    rep.check(a, "coordinates within valid ranges", int((~lat.between(-90, 90) | ~lon.between(-180, 180)).sum()), 0)
    rep.check(a, "each circuit has one coordinate pair",
              int((ra.groupby("circuit_id")[["circuit_lat", "circuit_lon"]].nunique().max(axis=1) > 1).sum()), 0)
    rep.check(a, "distinct circuits", ra["circuit_id"].nunique(), 77)
    rep.check(a, "distinct race names", ra["race_name"].nunique(), None)

    indy = ra[ra["race_name"] == "Indianapolis 500"]
    rep.check(a, "Indianapolis 500 races (exact race_name)", len(indy), EXPECTED["indianapolis_500_races"], "P-11")
    rep.check(a, "Indianapolis 500 seasons", f"{indy['season'].min()}-{indy['season'].max()}", "1950-1960", "P-11")
    usgp = ra[(ra["circuit_id"] == "indianapolis") & (ra["race_name"] != "Indianapolis 500")]
    rep.check(a, "US GP at circuit 'indianapolis' kept", len(usgp), EXPECTED["us_gp_at_indianapolis"], "P-11")
    rep.check(a, "eligible (non-Indy 500) races", len(ra) - len(indy), EXPECTED["eligible_races"], "P-11")
    rep.check(a, "races_policy has the same races and original values",
              bool(rp[ra.columns].equals(ra)), True)
    rep.check(a, "races_policy is_indianapolis_500 matches rule",
              int((flag(rp["is_indianapolis_500"]) != (rp["race_name"] == "Indianapolis 500")).sum()), 0, "P-11")
    rep.check(a, "races excluded from winner metrics (shared win)",
              int(flag(rp["exclude_from_winner_metrics"]).sum()), EXPECTED["shared_win_races"], "P-02")
    rep.anomaly(a, "indianapolis_500_race", "known_handled", "P-11", indy,
                lambda r: f"{r['race_name']} {r['season']}: excluded from model use, kept in data")


# ----------------------------------------------------------------------
# 2. Results + policy columns
# ----------------------------------------------------------------------
def validate_results(t, rep):
    res, rp, q = t["results"], t["results_policy"], t["qualifying"]
    a = "results"
    rep.check(a, "result rows", len(res), EXPECTED["result_rows"])
    for col in ["season", "round", "race_id", "driver_id", "constructor_id", "grid", "position",
                "position_text", "points", "laps", "status"]:
        rep.check(a, f"missing {col}", int(res[col].isna().sum()), 0)
    rep.check(a, "every result belongs to a known race",
              int((~res.set_index(RACE).index.isin(t["races"].set_index(RACE).index)).sum()), 0)
    rep.check(a, "race_id consistent with (season, round)",
              int((res["race_id"] != res["season"] + "_" + res["round"].str.zfill(2)).sum()), 0)
    pos, grid, laps, pts = num(res["position"]), num(res["grid"]), num(res["laps"]), num(res["points"])
    rep.check(a, "position numeric and >= 1", int((pos.isna() | (pos < 1)).sum()), 0)
    rep.check(a, "grid numeric and >= 0", int((grid.isna() | (grid < 0)).sum()), 0)
    rep.check(a, "laps numeric and >= 0", int((laps.isna() | (laps < 0)).sum()), 0)
    rep.check(a, "points numeric and >= 0", int((pts.isna() | (pts < 0)).sum()), 0)
    # Official numbering can skip positions (entries not in the data, e.g. non-qualifiers). Measured, not repaired.
    gap_races = []
    for race_id, g in res.assign(p=pos).groupby("race_id"):
        used = set(g["p"].astype(int))
        missing = sorted(set(range(1, int(g["p"].max()) + 1)) - used)
        if missing:
            gap_races.append({"race_id": race_id, "season": g["season"].iloc[0], "round": g["round"].iloc[0],
                              "driver_id": None, "missing": missing})
    rep.check(a, "races whose official position numbering skips positions", len(gap_races), None,
              detail="positions never used, mostly before withdrawn entries (F-007); positions not changed")
    rep.anomaly(a, "position_numbering_gap", "review", "P-10", pd.DataFrame(gap_races),
                lambda r: f"positions never used: {r['missing']} (official numbering kept)")
    rep.check(a, "positionText is a number or R/W/D",
              int((~res["position_text"].str.fullmatch(r"\d+|R|W|D")).sum()), 0, detail="F-003")
    numeric_text = res["position_text"].str.isdigit()
    rep.check(a, "numeric positionText == position", int((numeric_text & (res["position_text"] != res["position"])).sum()), 0)
    rep.check(a, "distinct raw status values (preserved)", res["status"].nunique(), EXPECTED["status_values"], "P-05")
    rep.check(a, "fully duplicated raw rows", int(res.duplicated().sum()), 0)
    rep.check(a, "car entry key (season, round, driver_id, driver_number) unique",
              int(res.duplicated(DRIVER_RACE + ["driver_number"]).sum()), 0, "P-01")
    dup = res[res.duplicated(DRIVER_RACE, keep=False)]
    rep.check(a, "rows of drivers with several entries in a race", len(dup), EXPECTED["multi_entry_rows"], "P-01")
    rep.check(a, "multi-entry (season, round, driver_id) groups", dup.groupby(DRIVER_RACE).ngroups,
              EXPECTED["multi_entry_groups"], "P-01")
    rep.check(a, "multi-entry seasons", f"{dup['season'].min()}-{dup['season'].max()}", "1950-1964", "P-01")
    shared = res[res.duplicated(RACE + ["position"], keep=False)]
    rep.check(a, "rows sharing a position (co-drivers)", len(shared), EXPECTED["shared_position_rows"], "P-02")
    rep.check(a, "races with shared positions", shared.groupby(RACE).ngroups, EXPECTED["shared_position_races"], "P-02")
    rep.check(a, "shared-position groups with one car number",
              int((shared.groupby(RACE + ["position"])["driver_number"].nunique() == 1).sum()),
              int(shared.groupby(RACE + ["position"]).ngroups), "P-02")
    rep.check(a, "grid = 0 rows (raw value preserved)", int((res["grid"] == "0").sum()), EXPECTED["grid_zero_rows"], "P-04")
    rep.anomaly(a, "multi_entry_driver", "known_handled", "P-01", dup,
                lambda r: f"car {r['driver_number']}, position {r['position']} ({r['position_text']}), {r['status']}")
    rep.anomaly(a, "shared_position", "known_handled", "P-02", shared,
                lambda r: f"position {r['position']} shared, car {r['driver_number']}, {r['constructor_id']}")

    # Policy file: original columns unchanged, and policy columns reproduce exactly from raw data
    a = "policies"
    rep.check(a, "results_policy original columns identical to results.csv", bool(rp[res.columns].equals(res)), True)
    recomputed = annotate_results(res, q)
    policy_cols = [c for c in recomputed.columns if c not in res.columns]
    mismatch = {c: int((recomputed[c].astype(str) != rp[c].astype(str)).sum()) for c in policy_cols}
    rep.check(a, "policy columns reproduce from raw data (src/policies.py)", sum(mismatch.values()), 0,
              "P-01..P-11", detail=str({k: v for k, v in mismatch.items() if v}) or "all 18 columns identical")
    prim = rp[flag(rp["is_primary_entry"])]
    rep.check(a, "exactly one primary entry per (season, round, driver)",
              int(prim.duplicated(DRIVER_RACE).sum() + (rp.groupby(DRIVER_RACE).ngroups - len(prim))), 0, "P-01")
    rep.check(a, "entry_count == rows per (season, round, driver)",
              int((num(rp["entry_count"]) != rp.groupby(DRIVER_RACE)["driver_id"].transform("size")).sum()), 0, "P-01")
    rep.check(a, "secondary entries", int((~flag(rp["is_primary_entry"])).sum()), EXPECTED["secondary_entries"], "P-01")
    rep.check(a, "races with a shared win", rp.loc[flag(rp["race_has_shared_win"]), "race_id"].nunique(),
              EXPECTED["shared_win_races"], "P-02")
    gs = rp["grid_status"]
    started = flag(rp["started"])
    laps = num(rp["laps"])
    rep.check(a, "pit-lane starts (grid 0, started, laps > 0)", int((gs == "pit_lane_start").sum()), EXPECTED["pit_lane_starts"], "P-04")
    rep.check(a, "pit_lane_start rule holds", int(((gs == "pit_lane_start") != ((rp["grid"] == "0") & started & (laps > 0))).sum()), 0, "P-04")
    rep.check(a, "grid unknown (grid 0, started, 0 laps)", int((gs == "unknown").sum()), EXPECTED["grid_unknown"], "P-04")
    rep.check(a, "non-starters have grid_status non_starter", int(((gs == "non_starter") != ~started).sum()), 0, "P-04")
    pop = rp[flag(rp["in_model_population"])]
    rep.check(a, "model population rows", len(pop), EXPECTED["model_population_rows"], "P-08")
    rep.check(a, "model population: one row per (season, round, driver)", int(pop.duplicated(DRIVER_RACE).sum()), 0, "P-08")
    rep.check(a, "model population: no Indianapolis 500 rows", int(flag(pop["is_indianapolis_500"]).sum()), 0, "P-11")
    rep.check(a, "model population: no non-starters", int((~flag(pop["started"])).sum()), 0, "P-08")
    rep.check(a, "model population: only primary entries", int((~flag(pop["is_primary_entry"])).sum()), 0, "P-01")
    rep.check(a, "model population races", pop["race_id"].nunique(), EXPECTED["eligible_races"], "P-08",
              detail="every eligible race has model rows")
    not_seq = []
    for race_id, g in pop.assign(p=num(pop["position"]).astype(int)).groupby("race_id"):
        distinct = sorted(set(g["p"]))
        if distinct != list(range(1, len(distinct) + 1)):
            not_seq.append({"race_id": race_id, "season": g["season"].iloc[0], "round": g["round"].iloc[0],
                            "driver_id": None, "positions": distinct})
    rep.check(a, "model-population races whose official positions are not 1..k", len(not_seq), None, "P-08/P-10",
              detail="caused by numbering gaps, secondary entries, shared positions or non-starters numbered in "
                     "between; relevant for the later position target (open question, F-007)")
    rep.anomaly(a, "model_population_positions_not_consecutive", "review", "P-08/P-10", pd.DataFrame(not_seq),
                lambda r: f"distinct official positions of model rows: {r['positions'][:3]}...{r['positions'][-3:]}")
    rep.check(a, "constructor_id unchanged (no merging)", bool(rp["constructor_id"].equals(res["constructor_id"])), True, "P-09")
    unknown_grid = rp[gs == "unknown"]
    rep.anomaly(a, "grid_zero_started_no_laps", "known_handled", "P-04", unknown_grid,
                lambda r: f"grid 0, {r['result_category']}, laps {r['laps']}: start position unknown")


# ----------------------------------------------------------------------
# 6-7. Status categories, DNF, ordering
# ----------------------------------------------------------------------
def validate_categories(t, rep):
    rp = t["results_policy"]
    a = "status_categories"
    detail = [result_category_detail(x, s, l) for x, s, l in zip(rp["position_text"], rp["status"], rp["laps"])]
    rep.check(a, "category recomputed from positionText/status/laps", int((pd.Series(detail) != rp["result_category_detail"]).sum()), 0, "P-05")
    counts = rp["result_category"].value_counts()
    for cat in ["finished", "classified_not_running", "started_not_classified", "disqualified", "did_not_start"]:
        rep.check(a, f"result_category = {cat}", int(counts.get(cat, 0)), EXPECTED[cat], "P-05/P-07")
    amb = rp[flag(rp["ambiguous_status"])]
    rep.check(a, "ambiguous rows treated as started", f"{len(amb)} rows, all started={bool(flag(amb['started']).all())}",
              f"{EXPECTED['ambiguous']} rows, all started=True", "P-07")
    dnf = rp["result_category"].map(is_dnf)
    rep.check(a, "DNF rows (started, not classified)", int(dnf.sum()), EXPECTED["dnf_rows"], "P-06")
    rep.check(a, "DNF rows have positionText R (or W with laps, P-07)",
              int((dnf & ~((rp["position_text"] == "R") | (rp["result_category_detail"] == "ambiguous_withdrawn_after_start"))).sum()), 0, "P-06")
    rep.check(a, "classified-not-running is NOT DNF", int((dnf & flag(rp["retired_while_classified"])).sum()), 0, "P-06")
    rep.check(a, "did_not_start rows have 0 laps", int((num(rp.loc[rp["result_category"] == "did_not_start", "laps"]) > 0).sum()), 0, "P-05")
    rep.check(a, "finished rows have a finished-type status",
              int((rp["result_category"].eq("finished") & ~rp["status"].str.match(FINISHED_STATUS)).sum()), 0, "P-05")
    rep.check(a, "no target column exists", int(sum(c.startswith("target") for c in rp.columns)), 0,
              detail="Step 7 creates no targets")
    rep.anomaly(a, "ambiguous_status", "known_handled", "P-07", amb,
                lambda r: f"{r['result_category_detail']} -> {r['result_category']} (status {r['status']}, laps {r['laps']})")
    dsq_numeric = rp[(rp["result_category"] == "disqualified") & rp["position_text"].str.isdigit()]
    rep.anomaly(a, "disqualified_with_numeric_positionText", "known_handled", "P-05", dsq_numeric,
                lambda r: f"status {r['status']}, positionText {r['position_text']}")

    a = "ordering"
    exc = rp[flag(rp["laps_order_exception"])]
    rep.check(a, "laps-order exception rows", len(exc), EXPECTED["order_exception_rows"], "P-10")
    rep.check(a, "laps-order exception races", exc["race_id"].nunique(), EXPECTED["order_exception_races"], "P-10")
    rep.check(a, "exception seasons", sorted(num(exc["season"]).astype(int).unique().tolist()), ORDER_EXCEPTION_SEASONS, "P-10")
    rep.check(a, "official positions unchanged", bool(rp["position"].equals(t["results"]["position"])), True, "P-10")
    rep.anomaly(a, "laps_order_exception", "known_handled", "P-10", exc,
                lambda r: f"position {r['position']} (R), laps {r['laps']}: official order kept")


# ----------------------------------------------------------------------
# 3-4. Qualifying and sprints
# ----------------------------------------------------------------------
def validate_qualifying(t, rep):
    q, ra, res = t["qualifying"], t["races"], t["results"]
    a = "qualifying"
    rep.check(a, "qualifying rows", len(q), EXPECTED["qualifying_rows"])
    for col in ["season", "round", "race_id", "driver_id", "constructor_id", "qualifying_position"]:
        rep.check(a, f"missing {col}", int(q[col].isna().sum()), 0)
    rep.check(a, "(season, round, driver_id) unique", int(q.duplicated(DRIVER_RACE).sum()), 0)
    qp = num(q["qualifying_position"])
    rep.check(a, "qualifying_position numeric >= 1", int((qp.isna() | (qp < 1)).sum()), 0)
    dup_pos = q[q.assign(p=qp).duplicated(RACE + ["p"], keep=False)]
    rep.check(a, "races with a repeated qualifying_position (source anomaly)", dup_pos["race_id"].nunique(), None,
              detail=f"{len(dup_pos)} rows; values not changed (F-007)")
    rep.anomaly(a, "repeated_qualifying_position", "review", "P-03", dup_pos,
                lambda r: f"qualifying position {r['qualifying_position']} shared; q1={r['q1']}, q2={r['q2']}, q3={r['q3']}")
    s = num(q["season"])
    rep.check(a, "qualifying seasons", f"{int(s.min())}-{int(s.max())}", "1994-2025")
    for col in ["q1", "q2", "q3"]:
        v = q[col].dropna()
        rep.check(a, f"{col} values are lap times (m:ss.sss or ss.sss)", int((~v.str.match(LAP_TIME)).sum()), 0)
    rep.check(a, "q2 present only from 2005", int((q["q2"].notna() & (s < 2005)).sum()), 0)
    rep.check(a, "q3 present only from 2006", int((q["q3"].notna() & (s < 2006)).sum()), 0)
    rep.check(a, "q3 present without q2", int((q["q3"].notna() & q["q2"].isna()).sum()), 0)
    no_q1_but_later = q[q["q1"].isna() & (q["q2"].notna() | q["q3"].notna())]
    rep.check(a, "q2/q3 present without q1", len(no_q1_but_later), None, detail="reported as anomaly for review")
    rep.anomaly(a, "later_session_time_without_q1", "review", "P-03", no_q1_but_later,
                lambda r: f"q1 missing, q2={r['q2']}, q3={r['q3']}, position {r['qualifying_position']}")
    q_races = set(zip(q["season"], q["round"]))
    no_q = ra[[(x, y) not in q_races for x, y in zip(ra["season"], ra["round"])]]
    rep.check(a, "races without qualifying (kept, P-03)", len(no_q), EXPECTED["races_without_qualifying"], "P-03")
    rep.check(a, "races without qualifying from 2003", int((num(no_q["season"]) >= 2003).sum()), 0, "P-03")
    rep.check(a, "qualifying values imputed", 0, 0, "P-03", detail="missing qualifying stays missing")


def validate_sprints(t, rep):
    sp, ra, res = t["sprints"], t["races"], t["results"]
    a = "sprints"
    rep.check(a, "sprint rows", len(sp), EXPECTED["sprint_rows"])
    rep.check(a, "sprint races", sp.groupby(RACE).ngroups, EXPECTED["sprint_races"])
    s = num(sp["season"])
    rep.check(a, "sprint seasons", f"{int(s.min())}-{int(s.max())}", "2021-2025")
    for col in ["season", "round", "race_id", "driver_id", "constructor_id", "position", "status"]:
        rep.check(a, f"missing {col}", int(sp[col].isna().sum()), 0)
    rep.check(a, "(season, round, driver_id) unique", int(sp.duplicated(DRIVER_RACE).sum()), 0)
    rep.check(a, "fully duplicated rows", int(sp.duplicated().sum()), 0)
    sched = set(map(tuple, ra.loc[ra["sprint_scheduled"] == "True", RACE].values))
    have = set(map(tuple, sp[RACE].drop_duplicates().values))
    rep.check(a, "sprint results only on scheduled sprint weekends", len(have - sched), 0)
    rep.check(a, "scheduled sprint weekends without sprint results", len(sched - have), 0)
    rep.check(a, "sprint drivers also in that race's results",
              len(set(zip(sp["season"], sp["round"], sp["driver_id"])) - set(zip(res["season"], res["round"], res["driver_id"]))), 0)
    rep.check(a, "sprint constructor == race constructor (same driver, race)",
              int(sp.merge(res.drop_duplicates(DRIVER_RACE)[DRIVER_RACE + ["constructor_id"]], on=DRIVER_RACE,
                           suffixes=("", "_race")).pipe(lambda m: (m["constructor_id"] != m["constructor_id_race"]).sum())), 0)


# ----------------------------------------------------------------------
# 5. Drivers and constructors
# ----------------------------------------------------------------------
def validate_entities(t, rep):
    dr, co, res = t["drivers"], t["constructors"], t["results"]
    a = "drivers"
    rep.check(a, "(season, driver_id) unique in season lists", int(dr.duplicated(["season", "driver_id"]).sum()), 0)
    rep.check(a, "unique drivers in results", res["driver_id"].nunique(), EXPECTED["unique_drivers"])
    ident = ["driver_first_name", "driver_last_name", "driver_dob"]
    rep.check(a, "driver_id with more than one identity (season lists)",
              int((dr.groupby("driver_id")[ident].nunique().max(axis=1) > 1).sum()), 0)
    rep.check(a, "driver_id with more than one identity (results)",
              int((res.groupby("driver_id")[ident].nunique().max(axis=1) > 1).sum()), 0)
    rep.check(a, "same person (name + dob) with several driver_ids",
              int((res.dropna(subset=ident).groupby(ident)["driver_id"].nunique() > 1).sum()), 0)
    code_mixed = res.groupby("driver_id")["driver_code"].apply(lambda s: s.isna().any() and s.notna().any())
    rep.check(a, "driver_code present in some rows but not others", int(code_mixed.sum()), 0,
              detail="code is a driver-level attribute")
    no_dob = dr[dr["driver_dob"].isna()]
    rep.check(a, "season-list rows without date of birth", len(no_dob), None)
    rep.anomaly(a, "driver_without_dob_or_nationality", "review", "", no_dob.assign(round=None, race_id=None),
                lambda r: f"{r['driver_first_name']} {r['driver_last_name']}: dob/nationality missing in API")

    a = "constructors"
    rep.check(a, "(season, constructor_id) unique in season lists", int(co.duplicated(["season", "constructor_id"]).sum()), 0)
    rep.check(a, "unique constructors in results", res["constructor_id"].nunique(), EXPECTED["unique_constructors"], "P-09")
    rep.check(a, "constructor_id -> one name", int((res.groupby("constructor_id")["constructor_name"].nunique() > 1).sum()), 0, "P-09")
    rep.check(a, "constructor name -> one id", int((res.groupby("constructor_name")["constructor_id"].nunique() > 1).sum()), 0, "P-09")
    seasons = res.assign(s=num(res["season"])).groupby("constructor_id")["s"].apply(lambda s: sorted(s.unique()))
    gaps = seasons[seasons.map(lambda ss: max((b - a for a, b in zip(ss, ss[1:])), default=0) >= 5)]
    rep.check(a, "constructor IDs reused after a gap of 5+ seasons (kept, not merged)", len(gaps),
              EXPECTED["constructor_ids_reused_after_gap"], "P-09")
    for cid, ss in gaps.items():
        rep.anomaly(a, "constructor_id_reused_after_gap", "known_handled", "P-09",
                    detail=f"{cid}: seasons {ss[0]}-{ss[-1]}, largest gap "
                           f"{max(b - a for a, b in zip(ss, ss[1:]))}; history restarts only when the API ID changes")
    rep.check(a, "lineage / merged constructor columns", int(sum("lineage" in c for c in t["results_policy"].columns)), 0, "P-09")


# ----------------------------------------------------------------------
# 8. Weather
# ----------------------------------------------------------------------
def validate_weather(t, rep, config):
    wr, wh, ra = t["weather_race"], t["weather_hourly"], t["races"]
    w = config["weather"]
    a = "weather"
    rep.check(a, "weather_race rows == races", len(wr), EXPECTED["races"])
    rep.check(a, "weather_race race_ids == races race_ids", bool(set(wr["race_id"]) == set(ra["race_id"])), True)
    st = wr["weather_status"].value_counts()
    rep.check(a, "races with weather (status ok)", int(st.get("ok", 0)), EXPECTED["weather_ok"])
    rep.check(a, "excluded Indianapolis 500 races", int(st.get("excluded_indianapolis_500", 0)), EXPECTED["weather_excluded"], "P-11")
    rep.check(a, "failed weather requests", int(st.get("failed", 0)), EXPECTED["weather_failed"])
    rep.check(a, "races with quality errors", int(st.get("ok_with_errors", 0)), 0)
    rep.check(a, "model-usable weather races", int(flag(wr["model_use_allowed"]).sum()), EXPECTED["weather_model_use"])
    rep.check(a, "weather coverage of eligible races (%)",
              round(100 * st.get("ok", 0) / EXPECTED["eligible_races"], 2), 100.0)
    excl = wr[wr["weather_status"] == "excluded_indianapolis_500"]
    rep.check(a, "excluded races are exactly the Indianapolis 500", bool((excl["race_name"] == "Indianapolis 500").all() and
                                                                         len(excl) == EXPECTED["weather_excluded"]), True, "P-11")
    us = wr[(wr["circuit_id"] == "indianapolis") & (wr["race_name"] != "Indianapolis 500")]
    rep.check(a, "US GP 2000-2007 have weather", int((us["weather_status"] == "ok").sum()), EXPECTED["us_gp_at_indianapolis"], "P-11")

    s = num(wr["season"])
    src = wr["start_time_source"]
    rep.check(a, "official_utc starts (2007-2025)", int((src == "official_utc").sum()), EXPECTED["start_official_utc"], "P-12")
    rep.check(a, "p12_local_clock starts (2005-2006)", int((src == "p12_local_clock").sum()), EXPECTED["start_p12_local_clock"], "P-12")
    rep.check(a, "imputed 14:00 local starts (1950-2004)", int((src == "imputed_local_14h").sum()), EXPECTED["start_imputed"])
    rep.check(a, "start source matches season rule",
              int(((src == "official_utc") != (s >= 2007)).sum() + ((src == "p12_local_clock") != s.between(2005, 2006)).sum()
                  + ((src == "imputed_local_14h") != (s <= 2004)).sum()), 0, "P-12")
    rep.check(a, "p12_local_clock_applied flag == source", int((flag(wr["p12_local_clock_applied"]) != (src == "p12_local_clock")).sum()), 0, "P-12")
    rep.check(a, "imputed starts are 14:00 local", int((wr.loc[src == "imputed_local_14h", "start_local"].str[11:16] != "14:00").sum()), 0)
    p12 = wr[src == "p12_local_clock"]
    rep.check(a, "P-12 local start clock == raw race_time clock", int((p12["start_local"].str[11:16] != p12["race_time"].str[:5]).sum()), 0, "P-12")
    off = wr[src == "official_utc"]
    utc_from_raw = pd.to_datetime(off["race_date"] + "T" + off["race_time"].str.rstrip("Z"), utc=True)
    rep.check(a, "official starts == race_date + race_time (UTC)", int((pd.to_datetime(off["start_utc"], utc=True) != utc_from_raw).sum()), 0)
    rep.check(a, "raw race_time preserved (== races.csv)",
              bool(wr.set_index("race_id")["race_time"].reindex(ra["race_id"]).reset_index(drop=True).equals(ra["race_time"])), True)
    # recompute every start with the approved code and check UTC <-> local consistency
    local_clock = set(w["race_time_local_clock_seasons"])
    recomputed = [race_start(r.race_date, r.race_time if isinstance(r.race_time, str) else None, r.timezone,
                             w["assumed_local_start_hour_before_2005"], int(r.season) in local_clock)
                  for r in wr.itertuples()]
    rep.check(a, "start_utc reproduces from raw data + timezone",
              int(sum(u.isoformat() != x for (u, _, _), x in zip(recomputed, wr["start_utc"]))), 0, "P-12")
    rep.check(a, "start_local converts back to start_utc",
              int((pd.to_datetime(wr["start_local"].map(lambda v: pd.Timestamp(v).tz_convert("UTC").isoformat())) !=
                   pd.to_datetime(wr["start_utc"])).sum()), 0)
    rep.check(a, "season-level clock check flags (after P-12)", int(flag(wr["race_time_suspected_local"]).sum()), 0, "P-12")

    ok = wr[wr["weather_status"] == "ok"]
    gd = num(ok["grid_distance_km"])
    rep.check(a, "max weather-grid distance to circuit (km) <= limit", bool(gd.max() <= w["max_grid_distance_km"]), True,
              detail=f"max {gd.max():.2f} km, median {gd.median():.2f} km")

    a = "weather_hourly"
    rep.check(a, "hourly rows", len(wh), EXPECTED["weather_hourly_rows"])
    rep.check(a, "(race_id, time_utc) unique", int(wh.duplicated(["race_id", "time_utc"]).sum()), 0)
    rep.check(a, "hourly races == races with weather", bool(set(wh["race_id"]) == set(ok["race_id"])), True)
    rep.check(a, "no Indianapolis 500 hourly rows", int(wh["race_id"].isin(excl["race_id"]).sum()), 0, "P-11")
    lo, hi = w["hourly_table_offsets"]
    offsets = wh.assign(o=num(wh["hour_offset"])).groupby("race_id")["o"].apply(lambda o: sorted(o) == list(range(lo, hi + 1)))
    rep.check(a, f"every race has offsets {lo}..{hi} exactly once", int((~offsets).sum()), 0)
    t_utc = pd.to_datetime(wh["time_utc"], utc=True, format="%Y-%m-%dT%H:%MZ", errors="coerce")
    rep.check(a, "time_utc parses", int(t_utc.isna().sum()), 0)
    t_loc = wh["time_local"].map(lambda v: pd.Timestamp(v).tz_convert("UTC"))
    rep.check(a, "time_local converts to time_utc", int((pd.to_datetime(t_loc) != t_utc).sum()), 0)
    h0 = wh["race_id"].map(ok.set_index("race_id")["start_utc"].map(lambda v: pd.Timestamp(v).floor("h")))
    rep.check(a, "hour_offset == hours since start rounded down",
              int((((t_utc - pd.to_datetime(h0, utc=True)) / pd.Timedelta(hours=1)).round() != num(wh["hour_offset"])).sum()), 0)
    variables = config["api"]["openmeteo"]["hourly_variables"]
    window = flag(wh["in_race_window"]) | flag(wh["in_prestart_window"])
    rep.check(a, "missing values inside prestart/race windows", int(wh.loc[window, variables].isna().sum().sum()), 0)
    for var, (vlo, vhi) in PLAUSIBLE.items():
        v = num(wh[var])
        rep.check(a, f"{var} within [{vlo}, {vhi}]", int(((v < vlo) | (v > vhi)).sum()), 0,
                  detail=f"observed {v.min()} .. {v.max()}")
    # window summaries reproduce from the hourly table
    diffs = 0
    hourly = wh.assign(time_utc=t_utc, **{v: num(wh[v]) for v in variables})
    for race_id, g in hourly.groupby("race_id"):
        row = ok.set_index("race_id").loc[race_id]
        aligned = align(g[["time_utc"] + variables], pd.Timestamp(row["start_utc"]), row["timezone"])
        for window_name, prefix in [(w["race_window_hours"], "wx_race_"), (w["prestart_window_hours"], "wx_prestart_")]:
            for k, v in window_summary(aligned, window_name, prefix).items():
                stored = num(pd.Series([row[k]])).iloc[0]
                if not ((pd.isna(v) and pd.isna(stored)) or abs(float(v) - float(stored)) < 1e-6):
                    diffs += 1
    rep.check(a, "window summaries reproduce from hourly values", diffs, 0)
    rep.anomaly("weather", "imputed_start_time_limitation", "known_handled", "",
                detail=f"{EXPECTED['start_imputed']} races 1950-2004 use an ASSUMED 14:00 local start (no official times); "
                       "their weather windows are approximate. Reanalysis is not a pre-race forecast.")


# ----------------------------------------------------------------------
# 9. Cross-table consistency
# ----------------------------------------------------------------------
def keyset(df, cols):
    return set(map(tuple, df[cols].drop_duplicates().values))


def cross_tables(t):
    ra, res, q, sp, dr, co, wr, wh = (t[n] for n in ["races", "results", "qualifying", "sprints", "drivers",
                                                   "constructors", "weather_race", "weather_hourly"])
    ok_weather = wr[wr["weather_status"] == "ok"]
    relations = [
        # relation, left, right, key, left_df, right_df, expected left_only, expected right_only, note
        ("races <-> results", "races", "results", RACE, ra, res, 0, 0, "every race has results"),
        ("races <-> qualifying", "races", "qualifying", RACE, ra, q, EXPECTED["races_without_qualifying"], 0,
         "races without qualifying: 548 before 1994 + 83 in 1994-2002 (P-03)"),
        ("races <-> sprints", "races", "sprints", RACE, ra, sp, EXPECTED["races"] - EXPECTED["sprint_races"], 0,
         "sprints exist only for 24 weekends 2021-2025"),
        ("races <-> weather_race", "races", "weather_race", ["race_id"], ra, wr, 0, 0, "one weather row per race"),
        ("eligible races <-> weather (ok)", "races (non-Indy 500)", "weather_race (ok)", ["race_id"],
         ra[ra["race_name"] != "Indianapolis 500"], ok_weather, 0, 0, "P-11"),
        ("weather_race (ok) <-> weather_hourly", "weather_race (ok)", "weather_hourly", ["race_id"], ok_weather, wh, 0, 0, ""),
        ("results <-> drivers (season lists)", "results", "drivers", ["season", "driver_id"], res, dr, 0, 142,
         "right-only = season-list drivers without any record (F-004)"),
        ("results <-> constructors (season lists)", "results", "constructors", ["season", "constructor_id"], res, co, 0, None,
         "right-only = listed constructors without results"),
        ("qualifying <-> drivers", "qualifying", "drivers", ["season", "driver_id"], q, dr, 0, None, ""),
        ("qualifying <-> constructors", "qualifying", "constructors", ["season", "constructor_id"], q, co, 0, None, ""),
        ("sprints <-> drivers", "sprints", "drivers", ["season", "driver_id"], sp, dr, 0, None, ""),
        ("qualifying <-> results (race, driver)", "qualifying", "results", DRIVER_RACE, q,
         res[res.set_index(RACE).index.isin(q.set_index(RACE).index)], 7, 26,
         "left-only = failed to qualify (no result row); right-only = raced without a qualifying row (P-03)"),
        ("sprints <-> results (race, driver)", "sprints", "results", DRIVER_RACE, sp,
         res[res.set_index(RACE).index.isin(sp.set_index(RACE).index)], 0, None, ""),
        ("results <-> results_policy (rows)", "results", "results_policy", DRIVER_RACE + ["driver_number"], res,
         t["results_policy"], 0, 0, "policy file covers every result row"),
    ]
    rows = []
    for name, left, right, key, ldf, rdf, exp_l, exp_r, note in relations:
        lk, rk = keyset(ldf, key), keyset(rdf, key)
        l_only, r_only = len(lk - rk), len(rk - lk)
        ok = (exp_l is None or l_only == exp_l) and (exp_r is None or r_only == exp_r)
        rows.append({"relation": name, "left": left, "right": right, "key": ", ".join(key), "left_keys": len(lk),
                     "right_keys": len(rk), "left_only": l_only, "right_only": r_only,
                     "expected_left_only": "" if exp_l is None else exp_l,
                     "expected_right_only": "" if exp_r is None else exp_r,
                     "status": "PASS" if ok else "FAIL", "note": note})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 10. Missingness
# ----------------------------------------------------------------------
def missing_classes(name, col, df):
    """
    Return (structural, historical, legitimate, unexpected, rule) counts for missing values of one column.
      structurally missing   : not applicable by design (e.g. weather of excluded races)
      historically unavailable: the source does not provide the field for that era/driver
      legitimately absent    : absent because of what happened (no Q3 when knocked out, no time when retired)
      unexpected missing     : none of the above -> listed in raw_anomalies.csv
    """
    miss = df[col].isna()
    s = num(df["season"]) if "season" in df else pd.Series(np.nan, index=df.index)
    zero = pd.Series(False, index=df.index)
    structural, historical, legitimate = zero.copy(), zero.copy(), zero.copy()
    rule = ""
    # a field missing for every row of a season is not provided by the source for that season
    if "season" in df:
        whole_season = df.groupby("season")[col].transform(lambda v: v.isna().all())
    else:
        whole_season = zero

    if col == "race_time":
        historical, rule = s <= 2004, "no official start time before 2005"
    elif name == "weather_race" and (col.startswith("wx_") or col.startswith("grid_")):
        structural, rule = df["weather_status"] == "excluded_indianapolis_500", "not requested: Indianapolis 500 (P-11)"
    elif col in ("driver_code", "permanent_number", "driver_dob", "driver_nationality"):
        historical = zero.copy() if col in ("driver_dob", "driver_nationality") else pd.Series(True, index=df.index)
        rule = "driver-level attribute not provided by the API for this driver" if col in ("driver_code", "permanent_number") \
            else "should always exist"
    elif col in ("time_millis", "time_text"):
        cat = df["result_category"] if "result_category" in df else pd.Series(
            [result_category_detail(x, y, z) for x, y, z in zip(df["position_text"], df["status"], df["laps"])], index=df.index)
        legitimate = (cat != "finished") | df["status"].str.match(r"^(Lapped|\+\d+ Laps?)$")
        historical = whole_season & ~legitimate
        rule = "race time only for cars finishing on the lead lap"
    elif col.startswith("fastest_lap"):
        laps = num(df["laps"])
        historical = (s < 2004) | whole_season
        legitimate = ~historical & ((laps <= 1) | df["position_text"].isin(["W"]))
        rule = "fastest lap data only from 2004 (and not for every field/season); none if <= 1 lap"
    elif col == "q1":
        legitimate, rule = pd.Series(True, index=df.index), "driver set no Q1 time"
    elif col in ("q2", "q3"):
        first = 2005 if col == "q2" else 2006
        historical = s < first
        legitimate = ~historical
        rule = f"{col} session exists only from {first}; afterwards absent when knocked out earlier"
    structural, historical, legitimate = structural & miss, historical & miss & ~structural, legitimate & miss & ~structural & ~historical
    unexpected = miss & ~structural & ~historical & ~legitimate
    return int(structural.sum()), int(historical.sum()), int(legitimate.sum()), int(unexpected.sum()), rule, unexpected


def missingness(t, rep):
    rows = []
    base_of = {"results_policy": "results", "races_policy": "races"}
    for name in TABLES:
        df = t[name]
        for col in df.columns:
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            if name in base_of and col in t[base_of[name]].columns:
                continue  # identical copy of the base table's column (validated separately)
            st, hi, le, un, rule, un_mask = missing_classes(name, col, df)
            rows.append({"table": name, "column": col, "rows": len(df), "missing": n_miss,
                         "missing_pct": round(100 * n_miss / len(df), 2), "structurally_missing": st,
                         "historically_unavailable": hi, "legitimately_absent": le, "unexpected_missing": un,
                         "rule": rule})
            # drivers without dob/nationality are listed by validate_entities already
            if un and name in ("results", "qualifying", "sprints", "races", "weather_race"):
                rep.anomaly("missingness", f"unexpected_missing_{col}", "review", "", df[un_mask],
                            lambda r, c=col, n=name: f"{n}.{c} missing (status {r.get('status', '')}, laps {r.get('laps', '')})")
    return pd.DataFrame(rows, columns=["table", "column", "rows", "missing", "missing_pct", "structurally_missing",
                                       "historically_unavailable", "legitimately_absent", "unexpected_missing", "rule"])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(interim_dir, tables_dir, config):
    t = load(interim_dir)
    rep = Report()
    validate_races(t, rep)
    validate_results(t, rep)
    validate_categories(t, rep)
    validate_qualifying(t, rep)
    validate_sprints(t, rep)
    validate_entities(t, rep)
    validate_weather(t, rep, config)
    cross = cross_tables(t)
    for r in cross.itertuples():
        rep.check("cross_table", r.relation, f"left_only={r.left_only}, right_only={r.right_only}",
                  f"left_only={r.expected_left_only or 'any'}, right_only={r.expected_right_only or 'any'}"
                  if r.status == "PASS" else "see raw_cross_table_validation.csv", detail=r.note)
        rep.checks[-1]["status"] = r.status
    miss = missingness(t, rep)
    rep.check("missingness", "unexpected missing values (all tables)", int(miss["unexpected_missing"].sum()), None,
              detail="listed in raw_anomalies.csv for review")

    outputs = {
        "raw_validation_summary": pd.DataFrame(rep.checks, columns=["area", "check", "expected", "observed", "status",
                                                                    "policy", "detail"]),
        "raw_missingness": miss,
        "raw_cross_table_validation": cross,
        "raw_anomalies": pd.DataFrame(rep.anomalies, columns=["area", "anomaly", "severity", "policy", "season",
                                                              "round", "race_id", "driver_id", "detail"]),
    }
    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8")
    return outputs


def main():
    config = load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    tables = PROJECT_ROOT / config["paths"]["tables"]
    protected = [interim / f"{n}.csv" for n in TABLES] + \
                [PROJECT_ROOT / "data" / "raw" / "_manifest.csv", PROJECT_ROOT / "data" / "raw" / "openmeteo" / "_manifest.csv"]
    before = digest(protected)
    out = run(interim, tables, config)
    assert digest(protected) == before, "an input file changed - validation must be read-only"
    summary = out["raw_validation_summary"]
    print(summary["status"].value_counts().to_string())
    failed = summary[summary["status"] == "FAIL"]
    if len(failed):
        print("\nFAILED CHECKS:\n" + failed.to_string(index=False))
    print(f"\nanomalies: {len(out['raw_anomalies'])} "
          f"({out['raw_anomalies']['severity'].value_counts().to_dict()}); inputs unchanged; no network used")


if __name__ == "__main__":
    main()
