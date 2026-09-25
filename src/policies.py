"""
Approved historical data policies P-01 ... P-11 (Step 5.6, docs/DECISIONS_LOG.md).

The policies are applied as ADDITIONAL, documented columns on copies of the
interim tables. Original API columns are never changed and no row is deleted:

    data/interim/results.csv  ->  data/interim/results_policy.csv  (all rows + policy columns)
    data/interim/races.csv    ->  data/interim/races_policy.csv    (all races + policy columns)

This is NOT target generation or feature engineering: no target_* columns,
no rolling/form features, no recoded grid numbers. The DNF rule (P-06) is
provided as the function is_dnf() for later use.

Run from the project root:
    python -m src.policies
"""

import hashlib

import pandas as pd

from src.config import PROJECT_ROOT, load_config

FINISHED_STATUS = r"^(Finished|Lapped|\+\d+ Laps?)$"
DSQ_STATUSES = {"Disqualified", "Excluded", "Underweight"}
INDIANAPOLIS_500 = "Indianapolis 500"  # P-11: exact race name, never circuit_id
RACE_KEY = ["season", "round"]
DRIVER_RACE_KEY = ["season", "round", "driver_id"]


# ----------------------------------------------------------------------
# P-05 / P-06 / P-07: result categories (derived; raw status and positionText are kept)
# ----------------------------------------------------------------------
def result_category_detail(position_text, status, laps):
    """
    Detailed category of one result row from positionText + status + laps.
    Status alone is not enough (e.g. "Engine" occurs as R, W and numeric).
    Returns one of: finished, classified_not_running, started_not_classified,
    disqualified, did_not_start, ambiguous_withdrawn_after_start,
    ambiguous_classified_withdrawn, ambiguous_classified_not_classified, unknown.
    """
    text = "" if pd.isna(position_text) else str(position_text)
    status = "" if pd.isna(status) else str(status)
    laps_done = pd.to_numeric(laps, errors="coerce")
    if text == "D" or status in DSQ_STATUSES:
        return "disqualified"
    if text == "W" or status == "Did not start":
        return "ambiguous_withdrawn_after_start" if laps_done > 0 else "did_not_start"
    if text == "R":
        return "started_not_classified"
    if text.isdigit():
        if pd.Series([status]).str.match(FINISHED_STATUS).iloc[0]:
            return "finished"
        if status == "Withdrew":
            return "ambiguous_classified_withdrawn"
        if status == "Not classified":
            return "ambiguous_classified_not_classified"
        return "classified_not_running"
    return "unknown"


# P-07: the 3 ambiguous categories are treated as started, as the closest clear category
AMBIGUOUS_TREATED_AS = {
    "ambiguous_withdrawn_after_start": "started_not_classified",   # completed laps, not classified
    "ambiguous_classified_withdrawn": "classified_not_running",    # classified position
    "ambiguous_classified_not_classified": "classified_not_running",
}
STARTED_CATEGORIES = {"finished", "classified_not_running", "started_not_classified", "disqualified"}


def effective_category(detail):
    """Category used by later steps: ambiguous rows mapped per P-07."""
    return AMBIGUOUS_TREATED_AS.get(detail, detail)


def is_dnf(category):
    """P-06: primary DNF definition = started, not classified. Classified-but-not-running is NOT DNF."""
    return category == "started_not_classified"


# ----------------------------------------------------------------------
# P-04: grid = 0
# ----------------------------------------------------------------------
def grid_status(grid, started, laps):
    """
    given           grid > 0 and the driver started
    pit_lane_start  grid = 0, started, completed laps > 0
    unknown         grid = 0, started, 0 laps (start position cannot be known)
    non_starter     did not start (any grid value)
    The original grid value is never changed.
    """
    if not started:
        return "non_starter"
    if str(grid) != "0":
        return "given"
    return "pit_lane_start" if pd.to_numeric(laps, errors="coerce") > 0 else "unknown"


# ----------------------------------------------------------------------
# P-10: official order exceptions (non-classified rows ordered by laps)
# ----------------------------------------------------------------------
def laps_order_exception_rows(results):
    """
    Index labels of rows forming a laps inversion among positionText 'R' rows of a race:
    a row placed directly after another R row but with MORE laps. Both rows of each pair are returned.
    """
    flagged = set()
    r_rows = results[results["position_text"] == "R"].assign(
        _pos=pd.to_numeric(results["position"]), _laps=pd.to_numeric(results["laps"]))
    for _, g in r_rows.groupby(RACE_KEY, sort=False):
        g = g.sort_values("_pos")
        idx, laps = g.index.tolist(), g["_laps"].tolist()
        for i in range(len(idx) - 1):
            if laps[i] < laps[i + 1]:
                flagged.update([idx[i], idx[i + 1]])
    return flagged


# ----------------------------------------------------------------------
# Annotate results (all policies)
# ----------------------------------------------------------------------
POLICY_RESULT_COLUMNS = [
    "entry_count", "multi_entry_driver", "is_primary_entry",                       # P-01
    "shared_position", "shared_position_count", "race_has_shared_win",             # P-02
    "race_has_qualifying", "has_qualifying_row",                                   # P-03
    "grid_status", "pit_lane_start",                                               # P-04
    "result_category_detail", "result_category", "ambiguous_status", "started",    # P-05, P-07
    "retired_while_classified",                                                    # P-06
    "laps_order_exception",                                                        # P-10
    "is_indianapolis_500",                                                         # P-11
    "in_model_population",                                                         # P-08
]


def annotate_results(results, qualifying):
    """Return a copy of results with every original column unchanged plus the policy columns."""
    out = results.copy()

    # P-05/P-07: categories
    out["result_category_detail"] = [result_category_detail(t, s, l)
                                     for t, s, l in zip(out["position_text"], out["status"], out["laps"])]
    out["result_category"] = out["result_category_detail"].map(effective_category)
    out["ambiguous_status"] = out["result_category_detail"].str.startswith("ambiguous")
    out["started"] = out["result_category"].isin(STARTED_CATEGORIES)
    # P-06
    out["retired_while_classified"] = out["result_category"] == "classified_not_running"

    # P-01: several car entries of one driver in one race -> primary = best-placed started entry
    out["entry_count"] = out.groupby(DRIVER_RACE_KEY)["driver_id"].transform("size")
    out["multi_entry_driver"] = out["entry_count"] > 1
    order = out.assign(_not_started=~out["started"], _pos=pd.to_numeric(out["position"]),
                       _car=pd.to_numeric(out["driver_number"], errors="coerce"))
    order = order.sort_values(DRIVER_RACE_KEY + ["_not_started", "_pos", "_car"])
    primary = order.drop_duplicates(DRIVER_RACE_KEY, keep="first").index
    out["is_primary_entry"] = out.index.isin(primary)

    # P-02: several drivers credited with the same position (co-drivers of one car)
    out["shared_position_count"] = out.groupby(RACE_KEY + ["position"])["driver_id"].transform("size")
    out["shared_position"] = out["shared_position_count"] > 1
    shared_win_races = set(map(tuple, out.loc[out["shared_position"] & (out["position"] == "1"), RACE_KEY].values))
    out["race_has_shared_win"] = [(s, r) in shared_win_races for s, r in zip(out["season"], out["round"])]

    # P-03: qualifying availability (missing values stay missing)
    quali_races = set(zip(qualifying["season"], qualifying["round"]))
    quali_rows = set(zip(qualifying["season"], qualifying["round"], qualifying["driver_id"]))
    out["race_has_qualifying"] = [(s, r) in quali_races for s, r in zip(out["season"], out["round"])]
    out["has_qualifying_row"] = [(s, r, d) in quali_rows
                                 for s, r, d in zip(out["season"], out["round"], out["driver_id"])]

    # P-04: grid = 0
    out["grid_status"] = [grid_status(g, st, l) for g, st, l in zip(out["grid"], out["started"], out["laps"])]
    out["pit_lane_start"] = out["grid_status"] == "pit_lane_start"

    # P-10: ordering exceptions (flag only)
    out["laps_order_exception"] = out.index.isin(laps_order_exception_rows(out))

    # P-11 and P-08
    out["is_indianapolis_500"] = out["race_name"] == INDIANAPOLIS_500
    out["in_model_population"] = out["started"] & out["is_primary_entry"] & ~out["is_indianapolis_500"]
    return out[list(results.columns) + POLICY_RESULT_COLUMNS]


def annotate_races(races, annotated_results):
    """Race-level policy flags; every original race column unchanged, every race kept."""
    per_race = annotated_results.groupby(RACE_KEY).agg(
        result_rows=("driver_id", "size"),
        model_population_rows=("in_model_population", "sum"),
        race_has_qualifying=("race_has_qualifying", "any"),
        race_has_shared_win=("race_has_shared_win", "any"),
        race_has_shared_position=("shared_position", "any"),
        race_has_laps_order_exception=("laps_order_exception", "any"),
    ).reset_index()
    out = races.merge(per_race, on=RACE_KEY, how="left")
    out["is_indianapolis_500"] = out["race_name"] == INDIANAPOLIS_500
    out["exclude_from_winner_metrics"] = out["race_has_shared_win"].fillna(False).astype(bool)  # P-02
    return out


def policy_summary(annotated_results, annotated_races):
    r, ra = annotated_results, annotated_races
    rows = [
        ("P-01", "result rows kept (all entries)", len(r)),
        ("P-01", "rows of drivers with several entries in a race", int(r["multi_entry_driver"].sum())),
        ("P-01", "secondary entries (not primary)", int((~r["is_primary_entry"]).sum())),
        ("P-02", "rows with a shared position", int(r["shared_position"].sum())),
        ("P-02", "races excluded from winner metrics (shared win)", int(ra["exclude_from_winner_metrics"].sum())),
        ("P-03", "races without qualifying (kept)", int((~ra["race_has_qualifying"].fillna(False).astype(bool)).sum())),
        ("P-03", "result rows without a qualifying row (kept)", int((~r["has_qualifying_row"]).sum())),
        ("P-04", "grid = 0 rows (raw value kept)", int((r["grid"] == "0").sum())),
        ("P-04", "pit-lane starts", int(r["pit_lane_start"].sum())),
        ("P-04", "started with grid = 0 and 0 laps: grid unknown", int((r["grid_status"] == "unknown").sum())),
        ("P-05", "distinct raw status values kept", int(r["status"].nunique())),
        ("P-06", "DNF rows (started, not classified)", int(r["result_category"].map(is_dnf).sum())),
        ("P-06", "retired while classified (not DNF, flagged)", int(r["retired_while_classified"].sum())),
        ("P-07", "ambiguous rows treated as started (flagged)", int(r["ambiguous_status"].sum())),
        ("P-08", "started rows", int(r["started"].sum())),
        ("P-08", "non-starter rows (not in population)", int((~r["started"]).sum())),
        ("P-08", "model population rows", int(r["in_model_population"].sum())),
        ("P-09", "constructor_id values (unchanged, not merged)", int(r["constructor_id"].nunique())),
        ("P-10", "rows flagged as laps-order exceptions", int(r["laps_order_exception"].sum())),
        ("P-10", "races with a laps-order exception", int(ra["race_has_laps_order_exception"].fillna(False).sum())),
        ("P-11", "Indianapolis 500 races (excluded from population)", int(ra["is_indianapolis_500"].sum())),
        ("P-11", "Indianapolis 500 rows", int(r["is_indianapolis_500"].sum())),
    ]
    for category, n in r["result_category"].value_counts().items():
        rows.append(("P-05", f"result_category = {category}", int(n)))
    return pd.DataFrame(rows, columns=["policy", "measure", "value"])


def _digest(paths):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    config = load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    tables = PROJECT_ROOT / config["paths"]["tables"]
    inputs = [interim / "results.csv", interim / "races.csv", interim / "qualifying.csv"]
    before = _digest(inputs)

    results = pd.read_csv(inputs[0], dtype=str, keep_default_na=False, na_values=[""])
    races = pd.read_csv(inputs[1], dtype=str, keep_default_na=False, na_values=[""])
    qualifying = pd.read_csv(inputs[2], dtype=str, keep_default_na=False, na_values=[""])

    annotated = annotate_results(results, qualifying)
    annotated_races = annotate_races(races, annotated)
    summary = policy_summary(annotated, annotated_races)

    annotated.to_csv(interim / "results_policy.csv", index=False, encoding="utf-8")
    annotated_races.to_csv(interim / "races_policy.csv", index=False, encoding="utf-8")
    summary.to_csv(tables / "policy_application_summary.csv", index=False, encoding="utf-8")

    assert _digest(inputs) == before, "input files changed - this must never happen"
    print(summary.to_string(index=False))
    print(f"\nWrote results_policy.csv ({len(annotated)} rows), races_policy.csv ({len(annotated_races)} rows); "
          f"inputs unchanged.")


if __name__ == "__main__":
    main()
