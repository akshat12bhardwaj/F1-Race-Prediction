"""
Step 8: exploratory data analysis (READ-ONLY, descriptive).

Reads the validated interim tables (Step 7) and writes
    outputs/tables/eda_*.csv      one table per analysis (every figure has a table twin)
    outputs/figures/eda/*.png     figures

Rules (see docs/DECISIONS_LOG.md):
- Nothing in data/ is modified. Numbers are converted in memory only.
- No targets, no predictive features, no splits, no imputation.
- Outcome variables (position, points, laps, status, ...) are described, and
  labelled POST-RACE in eda_variable_roles.csv; they must never become inputs.
- Correlations are descriptive only, not feature selection.
- Weather is historical reanalysis, NOT a pre-race forecast. Start times:
  1950-2004 assumed 14:00 local; 2005-2006 P-12 local clock; 2007-2025 official UTC.

Run from the project root:   python -m src.eda
"""

import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.validate_raw import missing_classes  # noqa: E402

ERAS = [(1950, 1969), (1970, 1989), (1990, 2009), (2010, 2025)]
ERA_LABELS = [f"{a}-{b}" for a, b in ERAS]
REPRESENTATIVE_SEASONS = [1955, 1975, 1995, 2015, 2024]
CATEGORY_ORDER = ["finished", "classified_not_running", "started_not_classified", "disqualified", "did_not_start"]
WX_VARS = ["temp_mean_c", "humidity_mean_pct", "precip_sum_mm", "rain_sum_mm", "cloud_mean_pct",
           "wind_speed_mean_kmh", "wind_dir_mean_deg", "weather_code_max"]

# ---- chart style (dataviz reference palette; validated: categorical slots 1-5, ordinal blue ramp) ----
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ERA_COLORS = ["#86b6ef", "#3987e5", "#256abf", "#104281"]            # ordinal: older = lighter
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f0a0a0", "#b83232"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "font.family": "sans-serif", "lines.linewidth": 2,
    "figure.dpi": 110, "svg.hashsalt": "f1-eda",
})


# ======================================================================
# Loading (read-only)
# ======================================================================
def era_of(season):
    s = int(season)
    return next(lbl for (a, b), lbl in zip(ERAS, ERA_LABELS) if a <= s <= b)


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def load(interim_dir, tables_dir):
    names = ["races_policy", "results_policy", "qualifying", "sprints", "drivers", "constructors",
             "weather_race", "weather_hourly"]
    t = {n: _read(interim_dir / f"{n}.csv") for n in names}
    t["raw_anomalies"] = _read(tables_dir / "raw_anomalies.csv")
    return t


def prepare(t):
    """In-memory numeric working copies. Source frames are not changed."""
    num = lambda s: pd.to_numeric(s, errors="coerce")
    R = t["results_policy"].copy()
    for c in ["season", "round", "position", "grid", "laps", "points"]:
        R[c + "_n"] = num(R[c])
    for c in ["started", "in_model_population", "is_indianapolis_500", "is_primary_entry", "shared_position",
              "multi_entry_driver", "pit_lane_start", "retired_while_classified", "laps_order_exception"]:
        R[c] = R[c] == "True"
    R["era"] = R["season"].map(era_of)
    R["race_max_laps"] = R.groupby("race_id")["laps_n"].transform("max")
    R["dnf"] = R["result_category"] == "started_not_classified"      # P-06 definition (descriptive use only)
    P = R[R["in_model_population"]].copy()                              # P-08 population

    Q = t["qualifying"].copy()
    Q["season_n"], Q["qp"] = num(Q["season"]), num(Q["qualifying_position"])
    Q["era"] = Q["season"].map(era_of)

    W = t["weather_race"].copy()
    W["season_n"] = num(W["season"])
    W["era"] = W["season"].map(era_of)
    for c in W.columns:
        if c.startswith(("wx_", "grid_")):
            W[c] = num(W[c])
    W["ok"] = W["weather_status"] == "ok"

    H = t["weather_hourly"].copy()
    for c in ["hour_offset", "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "cloud_cover",
              "wind_speed_10m", "wind_direction_10m", "weather_code"]:
        H[c] = num(H[c])
    H = H.merge(W[["race_id", "start_time_source", "era"]], on="race_id", how="left")

    races = t["races_policy"].copy()
    races["season_n"] = num(races["season"])
    races["era"] = races["season"].map(era_of)
    races["eligible"] = races["is_indianapolis_500"] != "True"
    return {"R": R, "P": P, "Q": Q, "W": W, "H": H, "races": races, "sprints": t["sprints"],
            "drivers": t["drivers"], "constructors": t["constructors"], "anomalies": t["raw_anomalies"]}


def _r(df, digits=4):
    """Round floats for stable, readable output."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(digits)
    return out


# ======================================================================
# 1. Variable roles (leakage labelling)
# ======================================================================
ROLE_RULES = [
    # (table, columns, source_type, timing, note)
    ("results", ["season", "round", "race_id", "race_name", "race_date", "circuit_id", "circuit_name", "locality",
                 "country"], "raw", "pre-race (race metadata)", "known before the race"),
    ("results", ["race_time"], "raw", "pre-race (schedule)", "official start time; missing before 2005"),
    ("results", ["driver_id", "driver_code", "driver_first_name", "driver_last_name", "driver_dob",
                 "driver_nationality", "constructor_id", "constructor_name", "constructor_nationality",
                 "driver_number"], "raw", "pre-race (entry/identity)", "known before the race"),
    ("results", ["permanent_number"], "raw", "not historical", "driver's CURRENT number (F-001); do not use"),
    ("results", ["grid"], "raw", "pre-race at T0", "final starting grid is known at T0; 0 = see grid_status (P-04)"),
    ("results", ["position", "position_text", "points", "laps", "status", "time_millis", "time_text",
                 "fastest_lap_rank", "fastest_lap_lap", "fastest_lap_time", "fastest_lap_avg_speed",
                 "fastest_lap_speed_units"], "raw", "POST-RACE OUTCOME", "never a prediction input for this race"),
    ("results_policy", ["entry_count", "multi_entry_driver"], "policy-derived (P-01)", "pre-race (entries)",
     "entries are known before the race"),
    ("results_policy", ["is_primary_entry"], "policy-derived (P-01)", "POST-RACE (uses position)",
     "chooses best-placed entry: population filter only, never an input"),
    ("results_policy", ["shared_position", "shared_position_count", "race_has_shared_win"], "policy-derived (P-02)",
     "POST-RACE OUTCOME", "evaluation filter only"),
    ("results_policy", ["race_has_qualifying", "has_qualifying_row"], "policy-derived (P-03)", "pre-race at T0",
     "qualifying availability"),
    ("results_policy", ["grid_status", "pit_lane_start"], "policy-derived (P-04)",
     "pre-race fact, derived with post-race laps", "a pit-lane start is known at T0 but the flag is derived using laps; "
                                                   "recheck before any feature use"),
    ("results_policy", ["result_category_detail", "result_category", "ambiguous_status", "retired_while_classified"],
     "policy-derived (P-05/06/07)", "POST-RACE OUTCOME", "outcome categories; DNF = started_not_classified"),
    ("results_policy", ["started"], "policy-derived (P-07/P-08)", "population filter (T0)",
     "who took the start; defines the population, not an input"),
    ("results_policy", ["laps_order_exception"], "policy-derived (P-10)", "POST-RACE OUTCOME", "quality flag"),
    ("results_policy", ["is_indianapolis_500", "in_model_population"], "policy-derived (P-08/P-11)",
     "population filter", "row selection only"),
    ("qualifying", ["qualifying_position", "q1", "q2", "q3"], "raw", "pre-race at T0",
     "qualifying happens before the race; missing = missing (P-03)"),
    ("sprints", ["position", "points", "grid", "status", "laps"], "raw", "pre-race at T0 for the Grand Prix",
     "sprint is held before the GP (2021+); outcome of the sprint itself"),
    ("weather_race", [f"wx_prestart_{v}" for v in WX_VARS], "raw-aligned (Step 6)",
     "observed before T0 (reanalysis, not a forecast)", "1950-2004 at an assumed 14:00 local start"),
    ("weather_race", [f"wx_race_{v}" for v in WX_VARS], "raw-aligned (Step 6)",
     "DURING RACE (oracle-forecast proxy)", "not available at T0; only as a perfect-forecast proxy (blueprint 19.6)"),
    ("weather_race", ["start_time_source", "p12_local_clock_applied", "timezone", "grid_distance_km"],
     "raw-aligned / policy (P-12)", "metadata", "data-quality context"),
]


def variable_roles():
    rows = [{"table": tb, "column": c, "source_type": st, "timing": tm, "note": nt}
            for tb, cols, st, tm, nt in ROLE_RULES for c in cols]
    rows += [{"table": "eda (this step)", "column": c, "source_type": "descriptive EDA variable (in memory only)",
              "timing": tm, "note": nt} for c, tm, nt in [
        ("era", "pre-race", "season bucket for description"),
        ("race_max_laps", "POST-RACE OUTCOME", "laps of the leader; used only to describe laps completed"),
        ("position_change (grid - position)", "POST-RACE OUTCOME", "description only"),
        ("laps_completed_ratio", "POST-RACE OUTCOME", "description only")]]
    return pd.DataFrame(rows)


# ======================================================================
# 2. Overview and season / era counts
# ======================================================================
def overview(d):
    R, P, Q, W, races = d["R"], d["P"], d["Q"], d["W"], d["races"]
    drv_ids = set(R["driver_id"])
    listed_never = d["drivers"][~d["drivers"]["driver_id"].isin(drv_ids)]["driver_id"].nunique()
    rows = [
        ("seasons", races["season"].nunique(), "1950-2025"),
        ("races", len(races), "all collected"),
        ("Indianapolis 500 races (excluded, P-11)", int((~races["eligible"]).sum()), "1950-1960"),
        ("eligible races", int(races["eligible"].sum()), ""),
        ("result rows (all entries)", len(R), ""),
        ("model-population rows (P-08)", len(P), "started, primary entry, not Indy 500"),
        ("qualifying rows", len(Q), "1994-2025, gaps 1994-2002"),
        ("sprint rows", len(d["sprints"]), "2021-2025"),
        ("drivers with results", R["driver_id"].nunique(), ""),
        ("drivers in model population", P["driver_id"].nunique(), ""),
        ("drivers listed but never in any result", listed_never, "season lists only"),
        ("constructors (API ids, P-09)", R["constructor_id"].nunique(), "not merged"),
        ("circuits", races["circuit_id"].nunique(), ""),
        ("races with weather", int(W["ok"].sum()), "reanalysis"),
        ("weather coverage of eligible races (%)", round(100 * W["ok"].sum() / races["eligible"].sum(), 2), ""),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def season_counts(d):
    R, P, Q, W, races = d["R"], d["P"], d["Q"], d["W"], d["races"]
    g = races.groupby("season_n")
    out = pd.DataFrame({
        "races": g.size(), "eligible_races": g["eligible"].sum(),
        "sprint_weekends": g["sprint_scheduled"].apply(lambda s: (s == "True").sum()),
    })
    out["result_rows"] = R.groupby("season_n").size()
    out["model_rows"] = P.groupby("season_n").size()
    out["mean_starters_per_race"] = P.groupby(["season_n", "race_id"]).size().groupby("season_n").mean()
    out["drivers"] = P.groupby("season_n")["driver_id"].nunique()
    out["constructors"] = P.groupby("season_n")["constructor_id"].nunique()
    out["races_with_qualifying"] = Q.groupby("season_n")["race_id"].nunique()
    out["qualifying_rows"] = Q.groupby("season_n").size()
    out["races_with_weather"] = W[W["ok"]].groupby("season_n").size()
    out["start_time_source"] = W.groupby("season_n")["start_time_source"].agg(lambda s: s.mode().iloc[0])
    out["mean_leader_laps"] = P.groupby(["season_n", "race_id"])["race_max_laps"].first().groupby("season_n").mean()
    out = out.fillna({"races_with_qualifying": 0, "qualifying_rows": 0, "races_with_weather": 0})
    out["qualifying_race_coverage_pct"] = 100 * out["races_with_qualifying"] / out["races"]
    out.index.name = "season"
    return _r(out.reset_index().astype({"races_with_qualifying": int, "qualifying_rows": int, "races_with_weather": int}), 2)


def era_summary(d):
    R, P, Q, W, races = d["R"], d["P"], d["Q"], d["W"], d["races"]
    rows = []
    for era in ERA_LABELS:
        r, p, ra, w = R[R["era"] == era], P[P["era"] == era], races[races["era"] == era], W[(W["era"] == era) & W["ok"]]
        q = Q[Q["era"] == era]
        cls = p[p["result_category"].isin(["finished", "classified_not_running"])]
        rows.append({
            "era": era, "seasons": ra["season"].nunique(), "races": len(ra), "eligible_races": int(ra["eligible"].sum()),
            "model_rows": len(p), "mean_starters_per_race": p.groupby("race_id").size().mean(),
            "drivers_per_season": p.groupby("season")["driver_id"].nunique().mean(),
            "constructors_per_season": p.groupby("season")["constructor_id"].nunique().mean(),
            "dnf_rate_pct (started, not classified)": 100 * p["dnf"].mean(),
            "disqualified_pct": 100 * (p["result_category"] == "disqualified").mean(),
            "non_starter_pct (of all result rows)": 100 * (~r["started"]).mean(),
            "grid_zero_rows": int((r["grid_n"] == 0).sum()), "pit_lane_starts": int(r["pit_lane_start"].sum()),
            "multi_entry_rows (P-01)": int(r["multi_entry_driver"].sum()),
            "qualifying_race_coverage_pct": 100 * q["race_id"].nunique() / len(ra),
            "sprint_weekends": int((ra["sprint_scheduled"] == "True").sum()),
            "mean_leader_laps": p.groupby("race_id")["race_max_laps"].first().mean(),
            "mean_abs_position_change_classified": (cls["grid_n"] - cls["position_n"])[cls["grid_n"] > 0].abs().mean(),
            "pole_win_rate_pct": 100 * (p[p["grid_n"] == 1]["position_n"] == 1).mean(),
            "weather_coverage_pct": 100 * len(w) / ra["eligible"].sum(),
            "race_temp_mean_c": w["wx_race_temp_mean_c"].mean(),
            "races_with_race_precip_ge_0_1mm_pct": 100 * (w["wx_race_precip_sum_mm"] >= 0.1).mean(),
            "start_time_basis": ", ".join(sorted(w["start_time_source"].unique())),
        })
    return _r(pd.DataFrame(rows), 3)


def representative_seasons(d):
    P, Q, W, races = d["P"], d["Q"], d["W"], d["races"]
    rows = []
    for s in REPRESENTATIVE_SEASONS:
        p, ra, w = P[P["season_n"] == s], races[races["season_n"] == s], W[(W["season_n"] == s) & W["ok"]]
        cls = p[p["result_category"].isin(["finished", "classified_not_running"]) & (p["grid_n"] > 0)]
        rows.append({"season": s, "races": len(ra), "mean_starters_per_race": p.groupby("race_id").size().mean(),
                     "dnf_rate_pct": 100 * p["dnf"].mean(),
                     "pole_win_rate_pct": 100 * (p[p["grid_n"] == 1]["position_n"] == 1).mean(),
                     "mean_abs_position_change_classified": (cls["grid_n"] - cls["position_n"]).abs().mean(),
                     "races_with_qualifying": Q[Q["season_n"] == s]["race_id"].nunique(),
                     "races_with_race_precip_ge_0_1mm": int((w["wx_race_precip_sum_mm"] >= 0.1).sum()),
                     "constructors": p["constructor_id"].nunique(), "drivers": p["driver_id"].nunique()})
    return _r(pd.DataFrame(rows), 2)


# ======================================================================
# 3. Missing data
# ======================================================================
MISSING_FIELDS = {
    "results": ["grid", "position", "position_text", "status", "laps", "points", "time_millis",
                "fastest_lap_rank", "fastest_lap_lap", "fastest_lap_time", "fastest_lap_avg_speed"],
    "qualifying": ["qualifying_position", "q1", "q2", "q3"],
    "weather_race": ["race_time", "start_time_source", "p12_local_clock_applied"] +
                    [f"wx_prestart_{v}" for v in WX_VARS] + [f"wx_race_{v}" for v in WX_VARS],
}


def missingness(t):
    rows = []
    sources = {"results": t["results_policy"], "qualifying": t["qualifying"], "weather_race": t["weather_race"]}
    for name, fields in MISSING_FIELDS.items():
        df = sources[name]
        for col in fields:
            n = int(df[col].isna().sum())
            st, hi, le, un, rule, _ = missing_classes(name, col, df) if n else (0, 0, 0, 0, "", None)
            rows.append({"table": name, "field": col, "rows": len(df), "missing": n,
                         "missing_pct": round(100 * n / len(df), 3), "historically_unavailable": hi,
                         "structurally_absent": st, "legitimately_absent": le, "unexpected_missing": un, "rule": rule})
    return pd.DataFrame(rows)


def missingness_by_era(t):
    rows = []
    for name, df in [("results", t["results_policy"]), ("qualifying", t["qualifying"])]:
        era = df["season"].map(era_of)
        for col in MISSING_FIELDS[name]:
            for e in ERA_LABELS:
                part = df.loc[era == e, col]
                rows.append({"table": name, "field": col, "era": e, "rows": len(part),
                             "missing_pct": round(100 * part.isna().mean(), 2) if len(part) else np.nan})
    return pd.DataFrame(rows)


UNEXPECTED_EXPLANATION = {
    "fastest_lap": ("API omits fastest-lap data for some classified cars in some 2004+ races, and the sprint "
                    "fastest_lap_rank in some sprint seasons (source gap)",
                    "No: fastest-lap fields are POST-RACE outcomes and will never be inputs"),
    "time": ("one 'Finished' car without a race time in the API", "No: race time is a POST-RACE outcome"),
    "driver_dob": ("season-list drivers without date of birth in the API (little-known entrants)",
                   "Yes, if driver age becomes a candidate variable: leave missing + indicator"),
    "driver_nationality": ("same 14 drivers without nationality", "Only if nationality is ever used"),
}


def unexpected_missing(d):
    a = d["anomalies"]
    a = a[a["anomaly"].str.startswith("unexpected_missing") | (a["anomaly"] == "driver_without_dob_or_nationality")]
    rows = []
    for anomaly, g in a.groupby("anomaly", sort=True):
        if anomaly == "driver_without_dob_or_nationality":
            for field in ["driver_dob", "driver_nationality"]:
                exp, handle = UNEXPECTED_EXPLANATION[field]
                rows.append({"field": field, "table": "drivers", "values": len(g), "seasons": _span(g["season"]),
                             "races": 0, "example": g["detail"].iloc[0], "possible_explanation": exp,
                             "handle_later": handle})
            continue
        field = anomaly.replace("unexpected_missing_", "")
        for table, gt in g.groupby(g["detail"].str.split(".").str[0], sort=True):
            key = "fastest_lap" if field.startswith("fastest_lap") else "time"
            exp, handle = UNEXPECTED_EXPLANATION[key]
            rows.append({"field": field, "table": table, "values": len(gt), "seasons": _span(gt["season"]),
                         "races": gt["race_id"].nunique(), "example": gt["detail"].iloc[0],
                         "possible_explanation": exp, "handle_later": handle})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"field": "TOTAL", "table": "", "values": int(out["values"].sum()), "seasons": "", "races": "",
                         "example": "", "possible_explanation": "Step 7 count of unexpected missing values",
                         "handle_later": ""}
    return out


def _span(seasons):
    s = pd.to_numeric(seasons, errors="coerce").dropna().astype(int)
    return f"{s.min()}-{s.max()}" if len(s) else ""


# ======================================================================
# 4. Results, status, grid
# ======================================================================
def status_categories(d):
    R = d["R"]
    rows = []
    for era in ["all"] + ERA_LABELS:
        r = R if era == "all" else R[R["era"] == era]
        started = r[r["started"]]
        for cat in CATEGORY_ORDER:
            n = int((r["result_category"] == cat).sum())
            rows.append({"era": era, "result_category": cat, "rows": n, "share_of_all_rows_pct": 100 * n / len(r),
                         "share_of_starters_pct": (100 * (started["result_category"] == cat).mean()
                                                   if cat != "did_not_start" else np.nan)})
    return _r(pd.DataFrame(rows), 3)


def status_top(d, n=10):
    R = d["R"]
    rows = []
    for era in ERA_LABELS:
        vc = R[R["era"] == era]["status"].value_counts().head(n)
        rows += [{"era": era, "rank": i + 1, "status": s, "rows": int(c)} for i, (s, c) in enumerate(vc.items())]
    return pd.DataFrame(rows)


def dnf_by_season(d):
    P = d["P"]
    g = P.groupby("season_n")
    return _r(pd.DataFrame({"starters": g.size(), "dnf_rate_pct": 100 * g["dnf"].mean(),
                            "disqualified_pct": 100 * g["result_category"].apply(lambda s: (s == "disqualified").mean()),
                            "classified_not_running_pct": 100 * g["retired_while_classified"].mean()})
              .rename_axis("season").reset_index(), 3)


def finish_distribution(d):
    P = d["P"]
    rows = []
    for era in ERA_LABELS:
        p = P[P["era"] == era]
        vc = p["position_n"].value_counts().sort_index()
        rows += [{"era": era, "official_position": int(k), "rows": int(v), "share_pct": 100 * v / len(p)} for k, v in vc.items()]
    return _r(pd.DataFrame(rows), 3)


def position_change(d):
    P = d["P"]
    rows = []
    for era in ["all"] + ERA_LABELS:
        p = P if era == "all" else P[P["era"] == era]
        for label, part in [("classified (finished + classified_not_running), grid > 0",
                             p[p["result_category"].isin(["finished", "classified_not_running"]) & (p["grid_n"] > 0)]),
                            ("all starters, grid > 0 (official position incl. DNF order)", p[p["grid_n"] > 0])]:
            ch = part["grid_n"] - part["position_n"]
            rows.append({"era": era, "population": label, "rows": len(part), "mean_change": ch.mean(),
                         "median_change": ch.median(), "std_change": ch.std(), "gained_pct": 100 * (ch > 0).mean(),
                         "same_pct": 100 * (ch == 0).mean(), "lost_pct": 100 * (ch < 0).mean()})
    return _r(pd.DataFrame(rows), 3)


def points_laps(d):
    P = d["P"]
    rows = []
    for era in ERA_LABELS:
        p = P[P["era"] == era]
        ratio = p["laps_n"] / p["race_max_laps"]
        row = {"era": era, "starters": len(p), "scoring_share_pct": 100 * (p["points_n"] > 0).mean(),
               "mean_points_per_start": p["points_n"].mean(), "max_points_single_race": p["points_n"].max(),
               "median_laps_completed_ratio": ratio.median()}
        for cat in ["finished", "classified_not_running", "started_not_classified", "disqualified"]:
            row[f"mean_laps_ratio_{cat}"] = ratio[p["result_category"] == cat].mean()
        rows.append(row)
    return _r(pd.DataFrame(rows), 3)


def grid_distribution(d):
    R = d["R"]
    rows = []
    for era in ERA_LABELS:
        vc = R[R["era"] == era]["grid_n"].value_counts().sort_index()
        rows += [{"era": era, "grid": int(k), "rows": int(v)} for k, v in vc.items()]
    return pd.DataFrame(rows)


def grid_zero(d):
    R = d["R"]
    g0 = R[R["grid_n"] == 0]
    out = pd.crosstab(g0["era"], g0["grid_status"]).reindex(ERA_LABELS, fill_value=0)
    out["total_grid_zero"] = out.sum(axis=1)
    out["share_of_era_rows_pct"] = (100 * out["total_grid_zero"] / R.groupby("era").size().reindex(ERA_LABELS)).round(3)
    return out.reset_index()


def grid_vs_finish(d, max_grid=20):
    P = d["P"]
    rows = []
    for era in ["all"] + ERA_LABELS:
        p = P if era == "all" else P[P["era"] == era]
        for gpos in range(1, max_grid + 1):
            part = p[p["grid_n"] == gpos]
            if len(part) == 0:
                continue
            rows.append({"era": era, "grid": gpos, "starters": len(part),
                         "win_rate_pct": 100 * (part["position_n"] == 1).mean(),
                         "podium_rate_pct": 100 * (part["position_n"] <= 3).mean(),
                         "mean_official_position": part["position_n"].mean(),
                         "median_official_position": part["position_n"].median(),
                         "dnf_rate_pct": 100 * part["dnf"].mean()})
    return _r(pd.DataFrame(rows), 3)


def grid_finish_matrix(P, era, n=20):
    p = P[(P["era"] == era) & P["grid_n"].between(1, n)]
    m = pd.crosstab(p["grid_n"], p["position_n"].clip(upper=n + 1))
    m = m.reindex(index=range(1, n + 1), columns=range(1, n + 2), fill_value=0)
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0)


# ======================================================================
# 5-6. Drivers and constructors
# ======================================================================
def driver_stats(d):
    R, P, Q = d["R"], d["P"], d["Q"]
    g = P.groupby("driver_id")
    out = pd.DataFrame({
        "name": g.apply(lambda x: f"{x['driver_first_name'].iloc[0]} {x['driver_last_name'].iloc[0]}", include_groups=False),
        "first_season": g["season_n"].min(), "last_season": g["season_n"].max(),
        "seasons": g["season"].nunique(), "starts": g.size(),
        "wins": g["position_n"].apply(lambda s: int((s == 1).sum())),
        "podiums": g["position_n"].apply(lambda s: int((s <= 3).sum())),
        "dnfs": g["dnf"].sum(),
    })
    out["points_all_entries (P-01 sum)"] = R.groupby("driver_id")["points_n"].sum()
    out["result_rows_all_entries"] = R.groupby("driver_id").size()
    out["non_starts"] = R[~R["started"]].groupby("driver_id").size()
    out["races_with_multiple_entries"] = R[R["multi_entry_driver"]].groupby("driver_id")["race_id"].nunique()
    out["shared_position_rows"] = R[R["shared_position"]].groupby("driver_id").size()
    out["qualifying_rows"] = Q.groupby("driver_id").size()
    out["first_era"] = out["first_season"].map(era_of)
    out = out.fillna({"non_starts": 0, "races_with_multiple_entries": 0, "shared_position_rows": 0, "qualifying_rows": 0})
    out["dnf_rate_pct"] = 100 * out["dnfs"] / out["starts"]
    int_cols = ["first_season", "last_season", "non_starts", "races_with_multiple_entries", "shared_position_rows",
                "qualifying_rows", "dnfs"]
    out[int_cols] = out[int_cols].astype(int)
    return _r(out.sort_values(["starts", "wins"], ascending=False).rename_axis("driver_id").reset_index(), 3)


def driver_summary(d, drivers):
    R, P = d["R"], d["P"]
    listed_never = d["drivers"][~d["drivers"]["driver_id"].isin(set(R["driver_id"]))]["driver_id"].nunique()
    no_start = R.groupby("driver_id")["started"].any()
    starts = drivers["starts"]
    rows = [("drivers in model population", len(drivers)),
            ("drivers listed but never in any result", listed_never),
            ("drivers with result rows but never started", int((~no_start).sum())),
            ("median starts per driver", float(starts.median())),
            ("drivers with 1 start", int((starts == 1).sum())),
            ("drivers with >= 100 starts", int((starts >= 100).sum())),
            ("drivers with at least one win", int((drivers["wins"] > 0).sum())),
            ("drivers with a multi-entry race (P-01)", int((drivers["races_with_multiple_entries"] > 0).sum())),
            ("drivers with shared-position rows (P-02)", int((drivers["shared_position_rows"] > 0).sum()))]
    rows += [(f"new drivers first seen in {e}", int((drivers["first_era"] == e).sum())) for e in ERA_LABELS]
    return pd.DataFrame(rows, columns=["metric", "value"])


def constructor_stats(d):
    R, P = d["R"], d["P"]
    g = P.groupby("constructor_id")
    wins = P[P["position_n"] == 1].groupby("constructor_id")["race_id"].nunique()
    out = pd.DataFrame({
        "name": g["constructor_name"].first(), "first_season": g["season_n"].min(), "last_season": g["season_n"].max(),
        "seasons": g["season"].nunique(), "races_entered": g["race_id"].nunique(), "starts": g.size(),
        "race_wins": wins, "podium_finishes": g["position_n"].apply(lambda s: int((s <= 3).sum())),
        "dnfs": g["dnf"].sum(),
    })
    out["points_all_entries"] = R.groupby("constructor_id")["points_n"].sum()
    out = out.fillna({"race_wins": 0}).astype({"race_wins": int, "first_season": int, "last_season": int, "dnfs": int})
    out["dnf_rate_pct"] = 100 * out["dnfs"] / out["starts"]
    return _r(out.sort_values(["race_wins", "starts"], ascending=False).rename_axis("constructor_id").reset_index(), 3)


# ======================================================================
# 7. Qualifying
# ======================================================================
def qualifying_coverage(d):
    Q, races = d["Q"], d["races"]
    qs = Q.groupby("season_n")
    out = pd.DataFrame({"races": races.groupby("season_n").size(), "races_with_qualifying": qs["race_id"].nunique(),
                        "qualifying_rows": qs.size(),
                        "q1_present_pct": 100 * qs["q1"].apply(lambda s: s.notna().mean()),
                        "q2_present_pct": 100 * qs["q2"].apply(lambda s: s.notna().mean()),
                        "q3_present_pct": 100 * qs["q3"].apply(lambda s: s.notna().mean())})
    out = out[out.index >= 1990].fillna({"races_with_qualifying": 0, "qualifying_rows": 0})
    out["race_coverage_pct"] = 100 * out["races_with_qualifying"] / out["races"]
    out["format_in_data"] = np.select(
        [out["races_with_qualifying"] == 0, out["q3_present_pct"] > 0, out["q2_present_pct"] > 0],
        ["no qualifying data", "Q1/Q2/Q3 knock-out", "two sessions (Q1+Q2)"], "single session (Q1)")
    return _r(out.astype({"races_with_qualifying": int, "qualifying_rows": int}).rename_axis("season").reset_index(), 2)


def qualifying_vs_grid(d):
    P, Q = d["P"], d["Q"]
    m = P[P["season_n"] >= 2003].merge(Q[["season", "round", "driver_id", "qp"]], on=["season", "round", "driver_id"],
                                       how="left")
    m = m[m["qp"].notna()]
    diff = m["grid_n"] - m["qp"]
    buckets = pd.cut(diff, [-100, -3.5, -0.5, 0.5, 3.5, 10.5, 100],
                     labels=["grid 4+ better", "grid 1-3 better", "grid == qualifying", "grid 1-3 worse",
                             "grid 4-10 worse", "grid 11+ worse"])
    out = buckets.value_counts().reindex(buckets.cat.categories).rename("rows").to_frame()
    out.loc["pit-lane start (grid 0)"] = int((m["grid_n"] == 0).sum())
    out["share_pct"] = (100 * out["rows"] / len(m)).round(3)
    return out.rename_axis("grid_minus_qualifying_position").reset_index()


def qualifying_position_outcomes(d, max_pos=20):
    P, Q = d["P"], d["Q"]
    m = P[P["season_n"] >= 2003].merge(Q[["season", "round", "driver_id", "qp"]], on=["season", "round", "driver_id"])
    rows = []
    for qpos in range(1, max_pos + 1):
        part = m[m["qp"] == qpos]
        rows.append({"qualifying_position": qpos, "starters": len(part), "win_rate_pct": 100 * (part["position_n"] == 1).mean(),
                     "podium_rate_pct": 100 * (part["position_n"] <= 3).mean(),
                     "mean_official_position": part["position_n"].mean(), "dnf_rate_pct": 100 * part["dnf"].mean()})
    return _r(pd.DataFrame(rows), 3)


# ======================================================================
# 9. Weather
# ======================================================================
def weather_distribution(d):
    W = d["W"][d["W"]["ok"]]
    rows = []
    for window in ["prestart", "race"]:
        for v in WX_VARS:
            col = f"wx_{window}_{v}"
            for group, part in [("all", W)] + [(e, W[W["era"] == e]) for e in ERA_LABELS] + \
                               [(s, W[W["start_time_source"] == s]) for s in sorted(W["start_time_source"].unique())]:
                x = part[col]
                rows.append({"window": window, "variable": v, "group": group, "races": int(x.notna().sum()),
                             "mean": x.mean(), "std": x.std(), "min": x.min(), "p05": x.quantile(0.05),
                             "median": x.median(), "p95": x.quantile(0.95), "max": x.max()})
    return _r(pd.DataFrame(rows), 3)


def weather_prestart_vs_race(d):
    W = d["W"][d["W"]["ok"]]
    rows = []
    for v in ["temp_mean_c", "humidity_mean_pct", "precip_sum_mm", "cloud_mean_pct", "wind_speed_mean_kmh"]:
        a, b = W[f"wx_prestart_{v}"], W[f"wx_race_{v}"]
        diff = b - a
        rows.append({"variable": v, "races": int(diff.notna().sum()), "mean_race_minus_prestart": diff.mean(),
                     "median_diff": diff.median(), "mean_abs_diff": diff.abs().mean(),
                     "spearman_prestart_vs_race": a.corr(b, method="spearman")})
    wet_pre, wet_race = W["wx_prestart_precip_sum_mm"] >= 0.1, W["wx_race_precip_sum_mm"] >= 0.1
    rows.append({"variable": "precip >= 0.1 mm (wet window)", "races": len(W),
                 "mean_race_minus_prestart": np.nan, "median_diff": np.nan, "mean_abs_diff": np.nan,
                 "spearman_prestart_vs_race": np.nan,
                 "both_dry": int((~wet_pre & ~wet_race).sum()), "prestart_wet_only": int((wet_pre & ~wet_race).sum()),
                 "race_wet_only": int((~wet_pre & wet_race).sum()), "both_wet": int((wet_pre & wet_race).sum())})
    return _r(pd.DataFrame(rows), 4)


def race_outcome_summary(d):
    """Race-level descriptive outcome aggregates (POST-RACE, for description only)."""
    P = d["P"]
    cls = P[P["result_category"].isin(["finished", "classified_not_running"]) & (P["grid_n"] > 0)]
    out = pd.DataFrame({"starters": P.groupby("race_id").size(), "dnf_rate": P.groupby("race_id")["dnf"].mean(),
                        "mean_abs_position_change": (cls["grid_n"] - cls["position_n"]).abs().groupby(cls["race_id"]).mean(),
                        "pole_won": P[P["grid_n"] == 1].groupby("race_id")["position_n"].apply(lambda s: bool((s == 1).any()))})
    return out


def weather_wet_evidence(d):
    W = d["W"][d["W"]["ok"]].set_index("race_id")
    outcomes = race_outcome_summary(d)
    W = W.join(outcomes)
    rows = []
    for thr in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        race_wet = W["wx_race_precip_sum_mm"] > thr if thr == 0 else W["wx_race_precip_sum_mm"] >= thr
        pre_wet = W["wx_prestart_precip_sum_mm"] > thr if thr == 0 else W["wx_prestart_precip_sum_mm"] >= thr
        label = "> 0" if thr == 0 else f">= {thr}"
        rows.append({"race_window_precip_mm": label, "races": int(race_wet.sum()), "share_pct": 100 * race_wet.mean(),
                     "races_prestart_same_threshold": int(pre_wet.sum()),
                     "races_2007_2025 (official start)": int((race_wet & (W["start_time_source"] == "official_utc")).sum()),
                     "mean_dnf_rate_pct (these races)": 100 * W.loc[race_wet, "dnf_rate"].mean(),
                     "mean_dnf_rate_pct (other races)": 100 * W.loc[~race_wet, "dnf_rate"].mean(),
                     "mean_abs_position_change (these)": W.loc[race_wet, "mean_abs_position_change"].mean(),
                     "mean_abs_position_change (other)": W.loc[~race_wet, "mean_abs_position_change"].mean(),
                     "pole_win_pct (these)": 100 * W.loc[race_wet, "pole_won"].mean(),
                     "pole_win_pct (other)": 100 * W.loc[~race_wet, "pole_won"].mean()})
    code = W["wx_race_weather_code_max"]
    for label, mask in [("weather_code_max >= 51 (drizzle or worse)", code >= 51),
                        ("weather_code_max >= 61 (rain or worse)", code >= 61)]:
        rows.append({"race_window_precip_mm": label, "races": int(mask.sum()), "share_pct": 100 * mask.mean(),
                     "mean_dnf_rate_pct (these races)": 100 * W.loc[mask, "dnf_rate"].mean(),
                     "mean_dnf_rate_pct (other races)": 100 * W.loc[~mask, "dnf_rate"].mean(),
                     "mean_abs_position_change (these)": W.loc[mask, "mean_abs_position_change"].mean(),
                     "mean_abs_position_change (other)": W.loc[~mask, "mean_abs_position_change"].mean(),
                     "pole_win_pct (these)": 100 * W.loc[mask, "pole_won"].mean(),
                     "pole_win_pct (other)": 100 * W.loc[~mask, "pole_won"].mean()})
    return _r(pd.DataFrame(rows), 3)


def weather_wet_by_era(d, thresholds=(0.1, 0.5, 2.0)):
    """Wet vs dry comparison WITHIN each era (older eras had higher DNF rates, so pooling is confounded)."""
    W = d["W"][d["W"]["ok"]].set_index("race_id").join(race_outcome_summary(d))
    rows = []
    for era in ERA_LABELS:
        w = W[W["era"] == era]
        for thr in thresholds:
            wet = w["wx_race_precip_sum_mm"] >= thr
            rows.append({"era": era, "race_window_precip_mm": f">= {thr}", "wet_races": int(wet.sum()),
                         "dry_races": int((~wet).sum()), "wet_share_pct": 100 * wet.mean(),
                         "dnf_rate_pct_wet": 100 * w.loc[wet, "dnf_rate"].mean(),
                         "dnf_rate_pct_dry": 100 * w.loc[~wet, "dnf_rate"].mean(),
                         "mean_abs_position_change_wet": w.loc[wet, "mean_abs_position_change"].mean(),
                         "mean_abs_position_change_dry": w.loc[~wet, "mean_abs_position_change"].mean(),
                         "pole_win_pct_wet": 100 * w.loc[wet, "pole_won"].mean(),
                         "pole_win_pct_dry": 100 * w.loc[~wet, "pole_won"].mean()})
    return _r(pd.DataFrame(rows), 3)


def weather_hourly_profile(d):
    H = d["H"]
    rows = []
    for src in ["all"] + sorted(H["start_time_source"].dropna().unique()):
        h = H if src == "all" else H[H["start_time_source"] == src]
        g = h.groupby("hour_offset")
        part = pd.DataFrame({"races": g["race_id"].nunique(), "mean_temp_c": g["temperature_2m"].mean(),
                             "mean_precip_mm": g["precipitation"].mean(),
                             "share_hours_precip_gt_0_pct": 100 * g["precipitation"].apply(lambda s: (s > 0).mean()),
                             "mean_cloud_pct": g["cloud_cover"].mean(), "mean_wind_kmh": g["wind_speed_10m"].mean()})
        part.insert(0, "start_time_source", src)
        rows.append(part.reset_index())
    return _r(pd.concat(rows, ignore_index=True), 4)


# ======================================================================
# 11. Correlations (descriptive)
# ======================================================================
def correlations(d):
    P, Q, W = d["P"], d["Q"], d["W"]
    m = P.merge(Q[["season", "round", "driver_id", "qp"]], on=["season", "round", "driver_id"], how="left")
    rows = []

    def add(label, x, y, population, timing):
        ok = x.notna() & y.notna()
        rows.append({"pair": label, "population": population, "n": int(ok.sum()),
                     "spearman": x[ok].corr(y[ok], method="spearman"), "pearson": x[ok].corr(y[ok]),
                     "timing_of_variables": timing})
    g = m[m["grid_n"] > 0]
    add("grid vs official position", g["grid_n"], g["position_n"], "model population, grid > 0", "pre-race vs POST-RACE")
    for era in ERA_LABELS:
        e = g[g["era"] == era]
        add(f"grid vs official position ({era})", e["grid_n"], e["position_n"], "model population, grid > 0",
            "pre-race vs POST-RACE")
    qq = m[m["season_n"] >= 2003]
    add("qualifying position vs official position (2003-2025)", qq["qp"], qq["position_n"], "model population, has qualifying",
        "pre-race vs POST-RACE")
    add("qualifying position vs grid (2003-2025)", qq["qp"], qq["grid_n"].where(qq["grid_n"] > 0), "has qualifying, grid > 0",
        "pre-race vs pre-race")
    add("laps vs points", m["laps_n"], m["points_n"], "model population", "POST-RACE vs POST-RACE")
    add("laps vs official position", m["laps_n"], m["position_n"], "model population", "POST-RACE vs POST-RACE")
    ratio = m["laps_n"] / m["race_max_laps"]
    add("laps completed ratio vs official position", ratio, m["position_n"], "model population", "POST-RACE vs POST-RACE")
    race = W[W["ok"]].set_index("race_id").join(race_outcome_summary(d))
    for wx in ["wx_race_precip_sum_mm", "wx_prestart_precip_sum_mm", "wx_race_temp_mean_c", "wx_race_humidity_mean_pct",
               "wx_race_wind_speed_mean_kmh", "wx_race_cloud_mean_pct"]:
        timing = "during race (oracle proxy)" if "_race_" in wx else "before T0 (reanalysis)"
        add(f"{wx} vs race DNF rate", race[wx], race["dnf_rate"], "race level", f"{timing} vs POST-RACE")
        add(f"{wx} vs mean |position change|", race[wx], race["mean_abs_position_change"], "race level",
            f"{timing} vs POST-RACE")
    return _r(pd.DataFrame(rows), 4)


def correlation_matrix(d):
    P, Q, W = d["P"], d["Q"], d["W"]
    m = P[P["grid_n"] > 0].merge(Q[["season", "round", "driver_id", "qp"]], on=["season", "round", "driver_id"], how="left")
    m = m.merge(W[["race_id", "wx_race_temp_mean_c", "wx_race_precip_sum_mm", "wx_race_humidity_mean_pct",
                   "wx_race_wind_speed_mean_kmh"]], on="race_id", how="left")
    cols = {"grid_n": "grid", "qp": "qualifying_position", "position_n": "official_position (POST)",
            "laps_n": "laps (POST)", "points_n": "points (POST)", "wx_race_temp_mean_c": "race temp (DURING)",
            "wx_race_precip_sum_mm": "race precip (DURING)", "wx_race_humidity_mean_pct": "race humidity (DURING)",
            "wx_race_wind_speed_mean_kmh": "race wind (DURING)", "season_n": "season"}
    return m[list(cols)].rename(columns=cols).corr(method="spearman").round(4)


# ======================================================================
# 12. Anomalies
# ======================================================================
ANOMALY_NOTES = {
    "multi_entry_driver": ("1950s shared drives: one driver credited in several cars", "Handled by P-01 (primary entry)"),
    "shared_position": ("co-drivers of one car credited with one position", "Handled by P-02; skip in winner metrics"),
    "grid_zero_started_no_laps": ("started category but 0 laps: start position unknown", "Handled by P-04 (grid unknown)"),
    "indianapolis_500_race": ("Indianapolis 500 was a championship round 1950-1960", "Handled by P-11 (excluded)"),
    "ambiguous_status": ("'Withdrew' after laps, or classified 'Withdrew'/'Not classified'", "Handled by P-07"),
    "laps_order_exception": ("R rows with more laps placed below fewer laps (8 races)", "Handled by P-10 (official order kept)"),
    "disqualified_with_numeric_positionText": ("DSQ recorded with a numeric position code", "Handled by P-05 (status used)"),
    "constructor_id_reused_after_gap": ("brand IDs reused decades apart (e.g. mercedes 1954-55 / 2010+)",
                                        "Handled by P-09; use recent-window constructor history later"),
    "imputed_start_time_limitation": ("no official start times before 2005", "Limitation: 14:00 local assumed"),
    "position_numbering_gap": ("official numbering skips positions (entries not in the data)",
                               "YES: decide official position vs rank among starters when defining the target"),
    "model_population_positions_not_consecutive": ("gaps, secondary entries, shared positions, non-starters in between",
                                                   "YES: same target decision"),
    "repeated_qualifying_position": ("source errors in 2023-2025 qualifying positions",
                                     "YES if qualifying position is used: keep, consider grid as primary"),
    "later_session_time_without_q1": ("Q2/Q3 time without Q1 (source gap)", "Low: only if qualifying times are used"),
    "driver_without_dob_or_nationality": ("little-known entrants without dob in the API", "Only if driver age is used"),
}


def anomaly_summary(d):
    a = d["anomalies"].copy()
    a["kind"] = np.where(a["anomaly"].str.startswith("unexpected_missing"), "unexpected_missing_values", a["anomaly"])
    rows = []
    for kind, g in a.groupby("kind", sort=True):
        exp, handle = ANOMALY_NOTES.get(kind, ("see eda_unexpected_missing.csv", "No for post-race fields; see that table"))
        rows.append({"anomaly": kind, "severity": g["severity"].iloc[0], "policy": g["policy"].fillna("").iloc[0],
                     "rows": len(g), "races": g["race_id"].nunique(), "seasons": _span(g["season"]),
                     "possible_explanation": exp, "handle_later": handle})
    return pd.DataFrame(rows)


# ======================================================================
# Figures
# ======================================================================
def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def make_figures(d, T, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sc = T["eda_season_counts"]

    # 01 races and model rows per season
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(sc["season"], sc["races"], color=SLOT[0], width=0.8)
    axes[0].set(title="Races per season (1950-2025)", ylabel="races")
    axes[1].bar(sc["season"], sc["model_rows"], color=SLOT[0], width=0.8)
    axes[1].set(title="Model-population rows per season (starters, P-08)", ylabel="rows", xlabel="season")
    _save(fig, fig_dir / "eda_01_races_and_rows_per_season.png")

    # 02 drivers, constructors, starters per race
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, col, title, lab in [(axes[0], "drivers", "Drivers who started per season", "drivers"),
                                (axes[1], "constructors", "Constructors (API ids) per season", "constructors"),
                                (axes[2], "mean_starters_per_race", "Mean starters per race (grid size)", "starters")]:
        ax.plot(sc["season"], sc[col], color=SLOT[0])
        ax.set(title=title, ylabel=lab)
    axes[2].set_xlabel("season")
    _save(fig, fig_dir / "eda_02_drivers_constructors_grid_size.png")

    # 03 data availability: qualifying coverage, start-time source, sprint weekends
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(sc["season"], sc["qualifying_race_coverage_pct"], color=SLOT[0])
    axes[0].set(title="Races with qualifying data (%) - none before 1994, gaps 1996-2002", ylabel="% of races", ylim=(-5, 105))
    src_colors = {"imputed_local_14h": SLOT[1], "p12_local_clock": SLOT[2], "official_utc": SLOT[0]}
    src_labels = {"imputed_local_14h": "assumed 14:00 local (1950-2004)", "p12_local_clock": "P-12 local clock (2005-2006)",
                  "official_utc": "official UTC (2007-2025)"}
    for src, color in src_colors.items():
        m = sc["start_time_source"] == src
        axes[1].bar(sc.loc[m, "season"], sc.loc[m, "races_with_weather"], color=color, width=0.8, label=src_labels[src])
    axes[1].set(title="Races with weather, by start-time basis", ylabel="races")
    axes[1].legend(loc="upper left")
    axes[2].bar(sc["season"], sc["sprint_weekends"], color=SLOT[0], width=0.8)
    axes[2].set(title="Sprint weekends per season (from 2021)", ylabel="sprints", xlabel="season")
    _save(fig, fig_dir / "eda_03_data_availability_by_season.png")

    # 04 missingness by era heatmap
    mb = T["eda_missingness_by_era"]
    piv = mb.assign(label=mb["table"] + "." + mb["field"]).pivot(index="label", columns="era", values="missing_pct")
    piv = piv.reindex(columns=ERA_LABELS)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = SEQ.copy()
    cmap.set_bad(GRID)                       # eras without any rows (e.g. qualifying before 1994)
    im = ax.imshow(np.ma.masked_invalid(piv.values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            text = "no data" if np.isnan(v) else f"{v:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color=MUTED if np.isnan(v) else ("white" if v > 55 else INK))
    ax.set(title="Missing values by era (% of rows) - no imputation", xlabel="era")
    fig.colorbar(im, ax=ax, label="% missing", shrink=0.8)
    _save(fig, fig_dir / "eda_04_missingness_by_era.png")

    # 05 result categories by era (100% stacked)
    scat = T["eda_status_categories"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(ERA_LABELS))
    for i, cat in enumerate(CATEGORY_ORDER):
        vals = scat[(scat["result_category"] == cat) & (scat["era"] != "all")].set_index("era") \
            .reindex(ERA_LABELS)["share_of_all_rows_pct"].values
        ax.bar(ERA_LABELS, vals, bottom=bottom, color=SLOT[i], width=0.6, label=cat, edgecolor=SURFACE, linewidth=1.5)
        bottom += vals
    ax.set(title="Result categories by era (% of result rows; POST-RACE outcomes)", ylabel="% of rows", ylim=(0, 100))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1))   # same order as the stack
    _save(fig, fig_dir / "eda_05_result_categories_by_era.png")

    # 06 DNF rate per season
    dn = T["eda_dnf_by_season"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dn["season"], dn["dnf_rate_pct"], color=SLOT[0])
    ax.set(title="DNF rate per season (started, not classified - P-06)", ylabel="% of starters", xlabel="season", ylim=(0, None))
    _save(fig, fig_dir / "eda_06_dnf_rate_by_season.png")

    # 07 grid vs finish heatmaps by era
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, era in zip(axes, ERA_LABELS):
        mat = grid_finish_matrix(d["P"], era)
        im = ax.imshow(mat.values, cmap=SEQ, vmin=0, vmax=0.5, aspect="auto")
        ax.set_title(era)
        ax.set_xticks([0, 4, 9, 14, 20], ["1", "5", "10", "15", "21+"])
        ax.set_yticks([0, 4, 9, 14, 19], ["1", "5", "10", "15", "20"])
        ax.set_xlabel("official finishing position")
        ax.grid(False)
    axes[0].set_ylabel("grid position")
    fig.suptitle("Share of starters from each grid slot finishing in each position (POST-RACE, descriptive)",
                 fontsize=11, fontweight="bold", color=INK)
    fig.colorbar(im, ax=axes, label="share of starters from that grid slot", shrink=0.85)
    _save(fig, fig_dir / "eda_07_grid_vs_finish_heatmaps.png")

    # 08 win and podium rate by grid position, per era
    gv = T["eda_grid_vs_finish"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in [(axes[0], "win_rate_pct", "Win rate by grid position"),
                           (axes[1], "podium_rate_pct", "Podium rate by grid position")]:
        for era, color in zip(ERA_LABELS, ERA_COLORS):
            e = gv[(gv["era"] == era) & (gv["grid"] <= 10)]
            ax.plot(e["grid"], e[col], color=color, marker="o", markersize=5, label=era)
        ax.set(title=title + " (grid 1-10)", xlabel="grid position", ylabel="% of starters", xticks=range(1, 11))
    axes[0].legend(title="era")
    _save(fig, fig_dir / "eda_08_win_podium_rate_by_grid.png")

    # 09 position change histograms (classified finishers)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    P = d["P"]
    for ax, era, color in zip(axes, ERA_LABELS, ERA_COLORS):
        p = P[(P["era"] == era) & P["result_category"].isin(["finished", "classified_not_running"]) & (P["grid_n"] > 0)]
        ch = (p["grid_n"] - p["position_n"]).clip(-20, 20)
        ax.hist(ch, bins=np.arange(-20.5, 21.5, 1), color=color, weights=np.full(len(ch), 100 / max(len(ch), 1)))
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set(title=era, xlabel="positions gained (grid - finish)")
    axes[0].set_ylabel("% of classified finishers")
    fig.suptitle("Positions gained by classified finishers (POST-RACE; clipped at +/-20)", fontsize=11,
                 fontweight="bold", color=INK)
    _save(fig, fig_dir / "eda_09_position_change_by_era.png")

    # 10 grid = 0 by era and grid status
    gz = T["eda_grid_zero"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(gz))
    for i, st in enumerate(["non_starter", "pit_lane_start", "unknown"]):
        vals = gz[st].values if st in gz else np.zeros(len(gz))
        ax.bar(gz["era"], vals, bottom=bottom, color=SLOT[i], width=0.6, label=st, edgecolor=SURFACE, linewidth=1.5)
        bottom += vals
    ax.set(title="grid = 0 rows by era and grid_status (P-04; raw value kept)", ylabel="rows")
    ax.legend()
    _save(fig, fig_dir / "eda_10_grid_zero_by_era.png")

    # 11 qualifying Q1/Q2/Q3 coverage and qualifying-vs-grid
    qc = T["eda_qualifying_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for i, q in enumerate(["q1", "q2", "q3"]):
        axes[0].plot(qc["season"], qc[f"{q}_present_pct"], color=SLOT[i], label=q.upper())
    axes[0].plot(qc["season"], qc["race_coverage_pct"], color=INK2, linewidth=1, label="races with any qualifying")
    axes[0].set(title="Qualifying availability 1990-2025", ylabel="% (of qualifying rows / of races)", xlabel="season",
                ylim=(-5, 105))
    axes[0].legend(loc="lower right")
    qg = T["eda_qualifying_vs_grid"]
    axes[1].barh(qg["grid_minus_qualifying_position"].astype(str), qg["share_pct"], color=SLOT[0])
    axes[1].invert_yaxis()
    axes[1].set(title="Grid vs qualifying position (2003-2025)", xlabel="% of starters with qualifying")
    fig.subplots_adjust(wspace=0.45)
    _save(fig, fig_dir / "eda_11_qualifying.png")

    # 12 drivers: starts distribution and top winners
    ds = T["eda_driver_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(ds["starts"], bins=[1, 2, 5, 10, 20, 50, 100, 200, 400], color=SLOT[0], edgecolor=SURFACE)
    axes[0].set_xscale("log")
    axes[0].set(title="Starts per driver (model population)", xlabel="starts (log scale)", ylabel="drivers")
    top = ds.sort_values(["wins", "starts"], ascending=False).head(15)
    axes[1].barh(top["name"], top["wins"], color=SLOT[0])
    axes[1].invert_yaxis()
    axes[1].set(title="Most race wins (official position 1; POST-RACE)", xlabel="wins")
    _save(fig, fig_dir / "eda_12_drivers.png")

    # 13 constructors: top by wins
    cs = T["eda_constructor_stats"].head(15)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(cs["constructor_id"] + " (" + cs["first_season"].astype(str) + "-" + cs["last_season"].astype(str) + ")",
            cs["race_wins"], color=SLOT[0])
    ax.invert_yaxis()
    ax.set(title="Race wins by constructor_id (API ids, not merged - P-09)", xlabel="races won")
    _save(fig, fig_dir / "eda_13_constructors.png")

    # 14 weather distributions (race window)
    W = d["W"][d["W"]["ok"]]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    specs = [("temp_mean_c", "temperature (°C)"), ("humidity_mean_pct", "relative humidity (%)"),
             ("precip_sum_mm", "precipitation (mm, 2 h)"), ("rain_sum_mm", "rain (mm, 2 h)"),
             ("cloud_mean_pct", "cloud cover (%)"), ("wind_speed_mean_kmh", "wind speed (km/h)"),
             ("wind_dir_mean_deg", "wind direction (°, from)"), ("weather_code_max", "max WMO weather code")]
    for ax, (v, label) in zip(axes.flat, specs):
        x = W[f"wx_race_{v}"]
        ax.hist(x, bins=30, color=SLOT[0], edgecolor=SURFACE, linewidth=0.5)
        ax.set(xlabel=label, ylabel="races")
        if v in ("precip_sum_mm", "rain_sum_mm"):
            ax.set_yscale("log")
            ax.set_ylabel("races (log)")
    fig.suptitle("Race-window weather, 1,138 races (reanalysis; DURING race, not a forecast)", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, fig_dir / "eda_14_weather_distributions.png")

    # 15 weather by era
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    data = [W.loc[W["era"] == e, "wx_race_temp_mean_c"].dropna() for e in ERA_LABELS]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.5, medianprops={"color": INK})
    for patch, c in zip(bp["boxes"], ERA_COLORS):
        patch.set_facecolor(c)
        patch.set_edgecolor(AXIS)
    axes[0].set_xticks(range(1, 5), ERA_LABELS)
    axes[0].set(title="Race-window temperature by era", ylabel="°C")
    es = T["eda_era_summary"]
    axes[1].bar(es["era"], es["races_with_race_precip_ge_0_1mm_pct"], color=ERA_COLORS, width=0.6)
    axes[1].set(title="Races with race-window precipitation >= 0.1 mm", ylabel="% of races with weather")
    _save(fig, fig_dir / "eda_15_weather_by_era.png")

    # 16 pre-start vs race window
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, v, label in [(axes[0], "temp_mean_c", "temperature (°C)"), (axes[1], "precip_sum_mm", "precipitation (mm)")]:
        x, y = W[f"wx_prestart_{v}"], W[f"wx_race_{v}"]
        ax.scatter(x, y, s=10, color=SLOT[0], alpha=0.5, edgecolors="none")
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=AXIS, linewidth=1)
        ax.set(xlabel=f"pre-start window {label}", ylabel=f"race window {label}", title=f"Pre-start vs race: {label}")
    _save(fig, fig_dir / "eda_16_prestart_vs_race_weather.png")

    # 17 wet evidence
    we = T["eda_weather_wet_evidence"]
    thr = we[we["race_window_precip_mm"].str.startswith((">", ">="))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(thr["race_window_precip_mm"], thr["share_pct"], color=SLOT[0], width=0.6)
    axes[0].set(title="Races at or above a race-window precipitation level", xlabel="precipitation in race window (mm)",
                ylabel="% of races")
    x = np.arange(len(thr))
    axes[1].bar(x - 0.18, thr["mean_dnf_rate_pct (these races)"], width=0.34, color=SLOT[0], label="races at/above level")
    axes[1].bar(x + 0.18, thr["mean_dnf_rate_pct (other races)"], width=0.34, color=SLOT[1], label="other races")
    axes[1].set_xticks(x, thr["race_window_precip_mm"])
    axes[1].set(title="Mean DNF rate (descriptive, all eras mixed)", xlabel="precipitation in race window (mm)",
                ylabel="% of starters")
    axes[1].legend()
    _save(fig, fig_dir / "eda_17_wet_threshold_evidence.png")

    # 18 hourly profile around the start
    hp = T["eda_weather_hourly_profile"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for src in ["imputed_local_14h", "p12_local_clock", "official_utc"]:
        h = hp[hp["start_time_source"] == src]
        axes[0].plot(h["hour_offset"], h["mean_temp_c"], color=src_colors[src], label=src_labels[src])
        axes[1].plot(h["hour_offset"], h["share_hours_precip_gt_0_pct"], color=src_colors[src], label=src_labels[src])
    for ax in axes:
        ax.axvspan(-3, -1, color=GRID, alpha=0.6)
        ax.axvspan(0, 2, color="#cde2fb", alpha=0.6)
        ax.set_xlabel("hours relative to start hour (shaded: pre-start -3..-1, race 0..2)")
    axes[0].set(title="Mean temperature around the start", ylabel="°C")
    axes[1].set(title="Hours with precipitation > 0", ylabel="% of race-hours")
    axes[0].legend(loc="lower center")
    _save(fig, fig_dir / "eda_18_hourly_profile.png")

    # 19 correlation matrix
    cm = T["eda_correlation_matrix"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(cm.index)), cm.index)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Spearman correlations (descriptive only; POST = outcome, DURING = race window)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman rho")
    _save(fig, fig_dir / "eda_19_correlation_matrix.png")


# ======================================================================
# Main
# ======================================================================
def build_tables(t):
    d = prepare(t)
    T = {
        "eda_variable_roles": variable_roles(),
        "eda_dataset_overview": overview(d),
        "eda_season_counts": season_counts(d),
        "eda_era_summary": era_summary(d),
        "eda_representative_seasons": representative_seasons(d),
        "eda_missingness": missingness(t),
        "eda_missingness_by_era": missingness_by_era(t),
        "eda_unexpected_missing": unexpected_missing(d),
        "eda_status_categories": status_categories(d),
        "eda_status_top_by_era": status_top(d),
        "eda_dnf_by_season": dnf_by_season(d),
        "eda_finish_position_distribution": finish_distribution(d),
        "eda_position_change": position_change(d),
        "eda_points_laps": points_laps(d),
        "eda_grid_distribution": grid_distribution(d),
        "eda_grid_zero": grid_zero(d),
        "eda_grid_vs_finish": grid_vs_finish(d),
        "eda_qualifying_coverage": qualifying_coverage(d),
        "eda_qualifying_vs_grid": qualifying_vs_grid(d),
        "eda_qualifying_position_outcomes": qualifying_position_outcomes(d),
        "eda_weather_distribution": weather_distribution(d),
        "eda_weather_prestart_vs_race": weather_prestart_vs_race(d),
        "eda_weather_wet_evidence": weather_wet_evidence(d),
        "eda_weather_wet_by_era": weather_wet_by_era(d),
        "eda_weather_hourly_profile": weather_hourly_profile(d),
        "eda_correlations": correlations(d),
        "eda_correlation_matrix": correlation_matrix(d),
        "eda_anomaly_summary": anomaly_summary(d),
    }
    T["eda_driver_stats"] = driver_stats(d)
    T["eda_driver_summary"] = driver_summary(d, T["eda_driver_stats"])
    T["eda_constructor_stats"] = constructor_stats(d)
    return d, T


def run(interim_dir, tables_dir, fig_dir=None, input_tables_dir=None):
    """input_tables_dir: where Step 7's raw_anomalies.csv is read from (default: tables_dir)."""
    t = load(interim_dir, input_tables_dir or tables_dir)
    d, T = build_tables(t)
    for name, df in T.items():
        df.to_csv(tables_dir / f"{name}.csv", index=(name == "eda_correlation_matrix"), encoding="utf-8")
    if fig_dir is not None:
        make_figures(d, T, fig_dir)
    return T


def main():
    config = load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    tables = PROJECT_ROOT / config["paths"]["tables"]
    figures = PROJECT_ROOT / config["paths"]["figures"] / "eda"
    inputs = sorted(interim.glob("*.csv")) + [tables / "raw_anomalies.csv"]
    digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before = digest()
    T = run(interim, tables, figures)
    assert digest() == before, "an input file changed - EDA must be read-only"
    print(f"EDA: {len(T)} tables, {len(list(figures.glob('*.png')))} figures; inputs unchanged")


if __name__ == "__main__":
    main()
