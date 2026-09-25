"""
Step 5.5: historical data policy analysis.

Reads the Step 5 interim tables (READ-ONLY) and quantifies eight historical
issues, proposing a policy for each. Nothing in data/interim is changed;
the proposed categories and rules are NOT applied to any dataset here.

Run from the project root:
    python -m src.policy_analysis

Writes to outputs/tables/:
    historical_policy_analysis.csv          one row per issue / sub-issue
    historical_policy_examples.csv          concrete example records
    historical_status_category_proposal.csv every (status, positionText type) with a proposed category
    historical_qualifying_coverage.csv      qualifying availability per season 1990-2025
"""

import pandas as pd

from src.config import PROJECT_ROOT, load_config
# The category rule now lives in src/policies.py (approved in Step 5.6); same function, one source.
from src.policies import result_category_detail as propose_result_category

KEY = ["season", "round", "driver_id"]
EXAMPLE_COLUMNS = ["issue_id", "sub_issue", "season", "round", "race_id", "race_name", "driver_id",
                   "constructor_id", "driver_number", "grid", "position", "position_text", "laps",
                   "status", "note"]
POLICY_COLUMNS = ["issue_id", "issue", "sub_issue", "affected_seasons", "affected_races", "affected_rows",
                  "findings", "proposed_policy", "consequence_for_ml_dataset", "decision_status"]


# Mapping of the proposed categories to the five groups asked for in Step 5.5
CATEGORY_GROUP = {
    "finished": "a) started and finished",
    "classified_not_running": "b) started, did not finish (classified)",
    "started_not_classified": "b) started, did not finish (not classified)",
    "disqualified": "c) disqualified",
    "did_not_start": "d) did not start",
    "ambiguous_withdrawn_after_start": "e) ambiguous",
    "ambiguous_classified_withdrawn": "e) ambiguous",
    "ambiguous_classified_not_classified": "e) ambiguous",
    "unknown": "e) ambiguous",
}
STARTED = {"finished", "classified_not_running", "started_not_classified", "disqualified"}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def load_interim(interim_dir):
    return {name: pd.read_csv(interim_dir / f"{name}.csv", dtype=str)
            for name in ["races", "results", "qualifying", "sprints", "drivers", "constructors"]}


def with_category(results):
    out = results.copy()
    out["proposed_category"] = [propose_result_category(t, s, l) for t, s, l in
                                zip(out["position_text"], out["status"], out["laps"])]
    return out


def span(seasons):
    s = sorted({int(x) for x in seasons})
    return f"{s[0]}-{s[-1]} ({len(s)} seasons)" if s else "none"


def n_races(df):
    return int(df[["season", "round"]].drop_duplicates().shape[0])


def policy(issue_id, issue, sub_issue, df, findings, proposed, consequence, seasons=None, races=None, rows=None):
    return {"issue_id": issue_id, "issue": issue, "sub_issue": sub_issue,
            "affected_seasons": seasons if seasons is not None else span(df["season"]),
            "affected_races": races if races is not None else n_races(df),
            "affected_rows": rows if rows is not None else len(df),
            "findings": findings, "proposed_policy": proposed,
            "consequence_for_ml_dataset": consequence, "decision_status": "PENDING REVIEW"}


def examples(df, issue_id, sub_issue, note, n=5):
    ex = df.head(n).copy()
    ex["issue_id"], ex["sub_issue"] = issue_id, sub_issue
    ex["note"] = note(ex) if callable(note) else note
    return ex.reindex(columns=EXAMPLE_COLUMNS)


# ----------------------------------------------------------------------
# 1. Shared drives
# ----------------------------------------------------------------------
def analyse_shared_drives(results):
    dup = results[results.duplicated(KEY, keep=False)]
    groups = dup.groupby(KEY)
    entry_unique = not results.duplicated(KEY + ["driver_number"]).any()
    shared = results[results.duplicated(["season", "round", "position"], keep=False)]
    sgroups = shared.groupby(["season", "round", "position"])
    same_car = int((sgroups["driver_number"].nunique() == 1).sum())
    shared_wins = shared[shared["position"] == "1"]
    overlap = int(dup.index.isin(shared.index).sum())

    rows = [
        policy("1", "Shared drives before 1965", "1a) same driver in several cars in one race", dup,
               f"{groups.ngroups} (season, round, driver_id) groups, sizes {groups.size().value_counts().to_dict()}. "
               f"Every group has a different car number per row (distinct car numbers per group: "
               f"{groups['driver_number'].nunique().value_counts().to_dict()}). Constructors per group: "
               f"{groups['constructor_id'].nunique().value_counts().to_dict()}. "
               f"(season, round, driver_id, driver_number) is unique across all {len(results)} rows: {entry_unique}.",
               "Keep every row in interim as a separate CAR ENTRY with entry key (season, round, driver_id, "
               "driver_number). Never deduplicate. In the model dataset (one row per driver per race) use the "
               "driver's PRIMARY entry = best official position; add n_entries and multi_entry_driver flag. "
               "Derived season points must SUM all entries (points are split between shared drives).",
               "40 early races get one model row per driver instead of 2-3; no information is lost because "
               "secondary entries stay in interim. Affects only 1950-1964, far from validation (2022) and test "
               "(2023-2025). Blueprint key (race_id, driver_id) stays valid for model rows."),
        policy("1", "Shared drives before 1965", "1b) several drivers credited with one car's position", shared,
               f"{sgroups.ngroups} (season, round, position) groups with 2+ drivers; {same_car} of them share the "
               f"SAME car number and constructor (co-drivers of one car). Shared wins (position 1): "
               f"{shared_wins[['season', 'round']].drop_duplicates().shape[0]} races. Shared podium positions: "
               f"{int((pd.to_numeric(shared.drop_duplicates(['season', 'round', 'position'])['position']) <= 3).sum())} groups. "
               f"{overlap} of these rows are also 1a duplicate-driver rows (same car-swap practice).",
               "Keep all co-driver rows. Flag shared_car_result on every row whose (race, position) is shared. "
               "Each co-driver keeps the official position (so two drivers can both be P1 in 3 races). "
               "Race-level evaluation that assumes exactly one winner must skip races with a shared win.",
               "'Exactly one winner per race' (blueprint 12.1) holds for all races from 1965; the 3 shared-win "
               "races are excluded only from race-level winner metrics, not from training rows."),
    ]
    ex = pd.concat([
        examples(dup.sort_values(KEY + ["driver_number"]), "1", "1a", "same driver, different car numbers", 6),
        examples(shared_wins, "1", "1b", "co-drivers of one car credited with the same position", 6),
    ])
    return rows, ex


# ----------------------------------------------------------------------
# 2. Qualifying coverage
# ----------------------------------------------------------------------
def analyse_qualifying(races, results, qualifying):
    seasons = [str(s) for s in range(1990, 2026)]
    cov = pd.DataFrame({
        "races": races.groupby("season").size(),
        "races_with_qualifying": qualifying.groupby("season")["round"].nunique(),
        "result_rows": results.groupby("season").size(),
        "qualifying_rows": qualifying.groupby("season").size(),
    }).reindex(seasons).fillna(0).astype(int)
    cov["race_coverage"] = (cov["races_with_qualifying"] / cov["races"]).round(3)
    cov["row_coverage"] = (cov["qualifying_rows"] / cov["result_rows"]).round(3)
    cov = cov.rename_axis("season").reset_index()

    merged = results.merge(qualifying[KEY].assign(has_qualifying=True), on=KEY, how="left")
    quali_races = set(zip(qualifying["season"], qualifying["round"]))
    in_quali_race = [(s, r) in quali_races for s, r in zip(merged["season"], merged["round"])]
    missing_in_quali_race = merged[pd.Series(in_quali_race, index=merged.index) & merged["has_qualifying"].isna()]
    gap = cov[(cov["season"].astype(int).between(1994, 2002)) & (cov["race_coverage"] < 1)]
    race_level = races.merge(pd.DataFrame(sorted(quali_races), columns=["season", "round"]).assign(q=1),
                             how="left", on=["season", "round"])
    no_q_races = race_level[race_level["q"].isna() & race_level["season"].astype(int).between(1994, 2002)]
    gap_rows = results.merge(no_q_races[["season", "round"]], on=["season", "round"])

    rows = [
        policy("2", "Qualifying coverage", "2a) seasons before 1994", results[results["season"].astype(int) < 1994],
               "No qualifying records at all for 1950-1993 (44 seasons).",
               "qualifying_position and times = missing; race-level qualifying_available = 0. Do not drop races. "
               "Grid (always present) remains the main starting-order feature.",
               "Any model using qualifying features must handle 100% missingness before 1994 (median/grid "
               "imputation inside the pipeline + indicator), or those features are restricted to the E9 "
               "modern training windows.",
               races=int((races["season"].astype(int) < 1994).sum())),
        policy("2", "Qualifying coverage", "2b) partial seasons 1994-2002", gap_rows,
               "Races with qualifying / races: " + ", ".join(
                   f"{r.season}: {r.races_with_qualifying}/{r.races}" for r in gap.itertuples()) +
               f". {len(no_q_races)} races (1994-2002) have no qualifying; every race from 2003 has it.",
               "Represent as missing + two indicators: race-level qualifying_available (race has any qualifying "
               "data) and row-level has_qualifying_row. Never drop these races. Do not impute across drivers "
               "from other races.",
               "Prevents silently losing 83 races; the indicator lets models learn that missingness is an era "
               "artefact, not a driver property. Qualifying-gap features (gap to pole) must be NaN, not 0.",
               seasons=span(no_q_races["season"]), races=len(no_q_races)),
        policy("2", "Qualifying coverage", "2c) drivers without a qualifying row in races that have qualifying",
               missing_in_quali_race,
               f"{len(missing_in_quali_race)} result rows (statuses {missing_in_quali_race['status'].value_counts().head(5).to_dict()}).",
               "Row-level has_qualifying_row = 0; qualifying_position missing (grid still available).",
               "Small; handled by the same missing-value rule."),
    ]
    ex = pd.concat([
        examples(gap_rows.drop_duplicates(["season", "round"]), "2", "2b", "race with results but no qualifying records", 5),
        examples(missing_in_quali_race, "2", "2c", "result row without a qualifying row", 5),
    ])
    return rows, ex, cov


# ----------------------------------------------------------------------
# 3. grid = 0
# ----------------------------------------------------------------------
def era(season):
    s = int(season)
    return "1950-1969" if s < 1970 else "1970-1989" if s < 1990 else "1990-2009" if s < 2010 else "2010-2025"


def analyse_grid_zero(cat_results):
    g0 = cat_results[cat_results["grid"] == "0"].copy()
    g0["era"] = g0["season"].map(era)
    laps = pd.to_numeric(g0["laps"])
    started = g0["proposed_category"].isin(STARTED)
    pit_lane = g0[started & (laps > 0)]
    no_lap = g0[started & (laps == 0)]
    non_start = g0[~started]
    by_era = pd.crosstab(g0["era"], g0["proposed_category"]).to_dict("index")
    rows = [
        policy("3", "grid = 0", "3a) grid 0 on non-starters and ambiguous rows", non_start,
               f"{len(non_start)} of {len(g0)} grid-0 rows are did_not_start or ambiguous. By era and category: {by_era}.",
               "grid 0 on a non-starter means 'no starting position'. These rows are excluded from model rows "
               "anyway (issue 5), so no recoding is needed.",
               "None for model rows; this explains why grid 0 is common before 1970 (143 rows, all non-starters)."),
        policy("3", "grid = 0", "3b) grid 0 with laps > 0 (started): pit-lane start", pit_lane,
               f"{len(pit_lane)} rows, eras {pit_lane['era'].value_counts().to_dict()}; categories "
               f"{pit_lane['proposed_category'].value_counts().to_dict()}.",
               "Interpret as PIT-LANE START (data-driven rule: started and completed laps, not era-based). Later "
               "(Step 8/11): grid_position = field_size + 1 and pit_lane_start = 1; keep original grid.",
               "Matches blueprint 13.2 for the modern era. The rule needs no era threshold because no pre-2010 "
               "grid-0 row completed a lap."),
        policy("3", "grid = 0", "3c) grid 0, started category, 0 laps", no_lap,
               f"{len(no_lap)} rows, eras {no_lap['era'].value_counts().to_dict()}, categories "
               f"{no_lap['proposed_category'].value_counts().to_dict()}.",
               "Treat grid as UNKNOWN (missing) + flag grid_unknown. Do not assume pit-lane start.",
               "Tiny; mostly disqualified/excluded entries that never completed a lap."),
    ]
    ex = pd.concat([examples(non_start, "3", "3a", "grid 0 non-starter", 4),
                    examples(pit_lane, "3", "3b", "grid 0 and laps > 0: pit-lane start", 4),
                    examples(no_lap, "3", "3c", "grid 0, started category, no laps", 4)])
    return rows, ex


# ----------------------------------------------------------------------
# 4. Status categories
# ----------------------------------------------------------------------
def status_category_table(cat_results, cat_sprints):
    both = pd.concat([cat_results.assign(source="race"), cat_sprints.assign(source="sprint")], ignore_index=True)
    text = both["position_text"].fillna("<missing>")
    both["position_text_type"] = text.where(~text.str.isdigit(), "numeric")
    both["season_int"] = both["season"].astype(int)
    table = (both.groupby(["source", "status", "position_text_type", "proposed_category"])
             .agg(records=("season", "size"), first_season=("season_int", "min"), last_season=("season_int", "max"))
             .reset_index())
    table["category_group"] = table["proposed_category"].map(CATEGORY_GROUP)
    return table.sort_values(["source", "records"], ascending=[True, False]).reset_index(drop=True)


def analyse_statuses(cat_results, status_table):
    race_tab = status_table[status_table["source"] == "race"]
    per_status = race_tab.groupby("status")["proposed_category"].nunique()
    rows = []
    for category, group in CATEGORY_GROUP.items():
        part = cat_results[cat_results["proposed_category"] == category]
        if part.empty:
            continue
        top = part["status"].value_counts().head(6).to_dict()
        rows.append(policy("4", "Status categories", f"{group}: {category}", part,
                           f"{part['status'].nunique()} distinct statuses, top {top}.",
                           "Keep original status and positionText unchanged; add proposed_category (later).",
                           CATEGORY_CONSEQUENCE[category]))
    rows.append(policy("4", "Status categories", "summary: status alone is not a category", cat_results,
                       f"{race_tab['status'].nunique()} race statuses; {int((per_status > 1).sum())} of them occur in "
                       f"more than one proposed category (e.g. Engine: R, W and numeric).",
                       "Category = f(positionText, status, laps) as in propose_result_category(); full mapping in "
                       "historical_status_category_proposal.csv.",
                       "DNF target (optional) should use the category, not the status text; proposed DNF = "
                       "started_not_classified only (blueprint 14: positionText R/N), with classified_not_running "
                       "kept as a separate flag. Needs approval."))
    ex = pd.concat([examples(cat_results[cat_results["proposed_category"] == c], "4", c, CATEGORY_GROUP[c], 3)
                    for c in CATEGORY_GROUP if (cat_results["proposed_category"] == c).any()])
    return rows, ex


CATEGORY_CONSEQUENCE = {
    "finished": "Model row; position target = official position; DNF = 0.",
    "classified_not_running": "Model row; keeps official classified position and points; proposed DNF = 0 "
                              "(blueprint), flag retired_while_classified = 1.",
    "started_not_classified": "Model row (driver started); position target = official position (ordered by laps, "
                              "issue 7); proposed DNF = 1.",
    "disqualified": "Model row (driver started); official position is at the back; DNF = 0, is_dsq = 1; "
                    "excluded from DNF-model training only (blueprint 14).",
    "did_not_start": "NOT a model row (never started; known at prediction time). Kept in interim; not counted "
                     "as a start in history features.",
    "ambiguous_withdrawn_after_start": "Proposed: model row as started_not_classified (it completed laps); "
                                       "needs approval. All are early-era rows.",
    "ambiguous_classified_withdrawn": "Proposed: model row as classified_not_running; needs approval.",
    "ambiguous_classified_not_classified": "Proposed: model row as classified_not_running; needs approval.",
    "unknown": "Excluded until investigated.",
}


# ----------------------------------------------------------------------
# 5. Absent drivers / populations
# ----------------------------------------------------------------------
def analyse_populations(cat_results, qualifying, drivers, sprints):
    raced = set(zip(cat_results["season"], cat_results["driver_id"])) | \
        set(zip(qualifying["season"], qualifying["driver_id"])) | set(zip(sprints["season"], sprints["driver_id"]))
    list_only = drivers[[(s, d) not in raced for s, d in zip(drivers["season"], drivers["driver_id"])]]
    never = set(list_only["driver_id"]) - set(cat_results["driver_id"])
    dnq = qualifying.merge(cat_results[KEY].assign(r=1), on=KEY, how="left")
    dnq = dnq[dnq["r"].isna()]
    started = cat_results[cat_results["proposed_category"].isin(STARTED)]
    non_start = cat_results[cat_results["proposed_category"] == "did_not_start"]
    ambiguous = cat_results[cat_results["proposed_category"].str.startswith("ambiguous")]
    rows = [
        policy("5", "Populations", "5a) in season driver list, never participated", list_only,
               f"{len(list_only)} driver-seasons ({list_only['driver_id'].nunique()} drivers); {len(never)} of them "
               f"never have a race result in any season.",
               "Not model rows. Keep drivers.csv unchanged (it is the API's season list, not a participation list).",
               "No rows are created; driver counts must come from results, not from drivers.csv.",
               races="n/a"),
        policy("5", "Populations", "5b) failed to qualify (qualifying row, no result row)", dnq,
               f"Observable only where qualifying exists (1994+): {len(dnq)} rows. Before 1994, DNQ entries have "
               f"NO record at all (they are invisible; part of the 793 'absent_between_appearances').",
               "Not model rows (never started). Do not invent rows for invisible DNQs.",
               "The model population is 'starters', so DNQs are correctly absent; field_size counts starters only."),
        policy("5", "Populations", "5c) started the race (result row, started category)", started,
               f"{len(started)} rows: {started['proposed_category'].value_counts().to_dict()}.",
               "THESE are the model rows (after the Indianapolis 500 exclusion, issue 8).",
               f"Model population before other exclusions: {len(started)} rows (+{len(ambiguous)} ambiguous rows if "
               f"approved)."),
        policy("5", "Populations", "5d) non-starters that have a result row", non_start,
               f"{len(non_start)} rows with category did_not_start (positionText W, 0 laps); "
               f"grid 0 on {int((non_start['grid'] == '0').sum())}, grid > 0 on {int((non_start['grid'] != '0').sum())}.",
               "Not model rows; keep in interim; the filter uses the category (positionText/status/laps), NOT grid.",
               "Removes about 1.3% of result rows from the model population; they are known before the race "
               "(withdrawn/DNS) so excluding them is not leakage."),
    ]
    ex = pd.concat([examples(list_only.assign(race_id=None), "5", "5a", "listed for the season, no records", 4),
                    examples(dnq, "5", "5b", "qualifying row but no race result", 5),
                    examples(non_start[non_start["grid"] != "0"], "5", "5d", "did not start but has a grid value", 4)])
    return rows, ex


# ----------------------------------------------------------------------
# 6. Constructor identity
# ----------------------------------------------------------------------
def analyse_constructors(results):
    res = results.assign(s=results["season"].astype(int))
    ids, names = res["constructor_id"].nunique(), res["constructor_name"].nunique()
    one_to_one = bool((res.groupby("constructor_id")["constructor_name"].nunique() == 1).all() and
                      (res.groupby("constructor_name")["constructor_id"].nunique() == 1).all())
    first = res.groupby("constructor_id")["s"].min()
    last = res.groupby("constructor_id")["s"].max()

    gaps = []
    for cid, g in res.groupby("constructor_id"):
        seasons = sorted(g["s"].unique())
        biggest = max((b - a for a, b in zip(seasons, seasons[1:])), default=0)
        if biggest >= 5:
            gaps.append({"constructor_id": cid, "gap": biggest, "first": seasons[0], "last": seasons[-1]})
    gaps = pd.DataFrame(gaps, columns=["constructor_id", "gap", "first", "last"])

    successions = []
    for old in last.index[(last >= 1980) & (last < 2025)]:
        drivers_old = set(res[(res["constructor_id"] == old) & (res["s"] == last[old])]["driver_id"])
        for new in first.index[first == last[old] + 1]:
            drivers_new = set(res[(res["constructor_id"] == new) & (res["s"] == last[old] + 1)]["driver_id"])
            if drivers_old & drivers_new:
                successions.append(f"{old}->{new} ({last[old] + 1})")
    rows = [
        policy("6", "Constructor identity", "6a) one ID = one name", results,
               f"{ids} constructor IDs and {names} names; strict 1:1 mapping: {one_to_one}. The API never renames an "
               f"ID; a rename appears as a NEW ID (e.g. alphatauri -> rb).",
               "Use API constructor_id as the historical identity; keep constructor_name as a separate label. "
               "Never merge IDs by name similarity.",
               "Constructor history features restart when a team is renamed (e.g. rb in 2024).",
               seasons=span(results["season"]), races=n_races(results), rows=len(results)),
        policy("6", "Constructor identity", "6b) brand IDs reused after long gaps",
               results[results["constructor_id"].isin(gaps["constructor_id"])],
               f"{len(gaps)} IDs have a gap of 5+ seasons, e.g. " + ", ".join(
                   f"{g.constructor_id} {g.first}-{g.last} (gap {g.gap})" for g in gaps.sort_values("gap", ascending=False).head(6).itertuples()) +
               ". One ID can cover unrelated organisations (e.g. mercedes 1954-55 and 2010-25).",
               "Keep the ID. Build constructor features from RECENT windows (last 5/10/20 races, previous season) "
               "so a decades-old block has no effect; do not use all-time expanding constructor averages.",
               "Avoids 1950s Mercedes results influencing 2010s Mercedes features."),
        policy("6", "Constructor identity", "6c) renames cannot be detected automatically", results,
               f"{len(successions)} 'ID ends, new ID starts next season with a shared driver' candidates, mixing real "
               f"renames and driver moves: " + ", ".join(successions[:12]) + ", ...",
               "Do not infer lineage automatically. Optional (blueprint COULD): a small MANUAL, documented lineage "
               "table for modern teams only, as a separate column constructor_lineage_id.",
               "Without lineage, renamed teams lose form history for their first races under the new ID "
               "(affects a few 2019-2024 rows near the validation/test period).",
               seasons="candidates 1980-2025", races="n/a", rows=len(successions)),
    ]
    ex = pd.DataFrame([{"issue_id": "6", "sub_issue": "6b", "constructor_id": g.constructor_id,
                        "note": f"seasons {g.first}-{g.last}, largest gap {g.gap}"}
                       for g in gaps.sort_values("gap", ascending=False).head(6).itertuples()] +
                      [{"issue_id": "6", "sub_issue": "6c", "note": c} for c in successions[:8]]
                      ).reindex(columns=EXAMPLE_COLUMNS)
    return rows, ex


# ----------------------------------------------------------------------
# 7. Retiree ordering
# ----------------------------------------------------------------------
def retiree_order_checks(results):
    """Per race: laps inversions among R rows (a later position with MORE laps) and R-above-classified cases."""
    res = results.assign(pos=pd.to_numeric(results["position"]), lp=pd.to_numeric(results["laps"]))
    out = []
    for (season, rnd), g in res.groupby(["season", "round"], sort=False):
        g = g.sort_values("pos")
        retired = g[g["position_text"] == "R"]
        classified = g[g["position_text"].str.isdigit()]
        laps_r = retired["lp"].tolist()
        inversions = sum(a < b for a, b in zip(laps_r, laps_r[1:]))
        above = bool(len(retired) and len(classified) and retired["pos"].min() < classified["pos"].max())
        out.append({"season": season, "round": rnd, "race_id": g["race_id"].iloc[0], "retired_rows": len(retired),
                    "adjacent_pairs": max(len(retired) - 1, 0), "laps_inversions": int(inversions),
                    "retired_above_classified": above})
    return pd.DataFrame(out)


def analyse_retiree_ordering(results):
    checks = retiree_order_checks(results)
    bad = checks[checks["laps_inversions"] > 0]
    above = checks[checks["retired_above_classified"]]
    rows = [
        policy("7", "Retiree ordering", "7a) non-classified (R) rows ordered by laps completed", bad,
               f"{int(checks['retired_rows'].gt(0).sum())} races have R rows; {int(checks['adjacent_pairs'].sum())} adjacent "
               f"R pairs; only {int(bad['laps_inversions'].sum())} inversions in {len(bad)} races "
               f"(seasons {sorted(bad['season'].astype(int).unique().tolist())}); none after 2009.",
               "Accept the official numeric position as a meaningful order for ALL started rows (it follows laps "
               "completed in >99.8% of pairs). Keep the exceptions flagged; do not re-sort.",
               "Supports blueprint 14: position regression target covers non-finishers too.",
               races=len(bad), rows=int(bad["laps_inversions"].sum())),
        policy("7", "Retiree ordering", "7b) R row placed above a classified row", above,
               f"{len(above)} races: {above['race_id'].tolist()}.",
               "Flag; do not change. Inspect during Step 7 validation.",
               "Negligible."),
    ]
    ex_rows = []
    res = results.assign(pos=pd.to_numeric(results["position"]))
    for race_id in bad["race_id"].head(3):
        g = res[(res["race_id"] == race_id) & (res["position_text"] == "R")].sort_values("pos")
        laps = pd.to_numeric(g["laps"]).tolist()
        idx = next(i for i in range(len(laps) - 1) if laps[i] < laps[i + 1])
        ex_rows.append(examples(g.iloc[idx: idx + 2], "7", "7a", "later position has more laps"))
    for race_id in above["race_id"].head(3):
        ex_rows.append(examples(res[res["race_id"] == race_id].sort_values("pos").tail(3), "7", "7b",
                                "bottom of classification"))
    return rows, pd.concat(ex_rows) if ex_rows else pd.DataFrame(columns=EXAMPLE_COLUMNS)


# ----------------------------------------------------------------------
# 8. Indianapolis
# ----------------------------------------------------------------------
def analyse_indianapolis(races, results):
    indy_races = races[races["race_name"] == "Indianapolis 500"]
    indy_rows = results[results["race_name"] == "Indianapolis 500"]
    same_circuit_other = races[(races["circuit_id"] == "indianapolis") & (races["race_name"] != "Indianapolis 500")]
    other_drivers = set(results[results["race_name"] != "Indianapolis 500"]["driver_id"])
    crossover = sorted(set(indy_rows["driver_id"]) & other_drivers)
    name_contains = races[races["race_name"].str.contains("Indianapolis", case=False)]["race_name"].unique().tolist()
    rows = [policy("8", "Indianapolis 500", "8a) exclusion rule", indy_rows,
                   f"race_name == 'Indianapolis 500' selects exactly {len(indy_races)} races (1950-1960) and "
                   f"{len(indy_rows)} rows ({indy_rows['driver_id'].nunique()} drivers). circuit_id 'indianapolis' is "
                   f"ALSO used by {len(same_circuit_other)} races: {same_circuit_other['race_name'].unique().tolist()} "
                   f"{same_circuit_other['season'].min()}-{same_circuit_other['season'].max()}. Race names containing "
                   f"'Indianapolis': {name_contains}. Drivers who also raced other GPs: {crossover}.",
                   "Exclude rows where race_name == 'Indianapolis 500' (exact match) from model rows AND from driver/"
                   "constructor/circuit history. Never filter by circuit_id. Keep the rows in interim.",
                   f"Removes {len(indy_rows)} rows / {len(indy_races)} races; the US GP 2000-2007 stays. "
                   f"Season race counts used for round-based features must be recomputed without these rounds.")]
    ex = pd.concat([examples(indy_rows.drop_duplicates("race_id"), "8", "8a", "Indianapolis 500 (exclude)", 3),
                    examples(results[results["race_id"].isin(same_circuit_other["race_id"])].drop_duplicates("race_id"),
                             "8", "8a", "US GP at the same circuit_id (keep)", 3)])
    return rows, ex


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(interim_dir, tables_dir):
    t = load_interim(interim_dir)
    results = with_category(t["results"])
    sprints = with_category(t["sprints"])

    status_table = status_category_table(results, sprints)
    parts = [analyse_shared_drives(results)]
    q_rows, q_ex, coverage = analyse_qualifying(t["races"], results, t["qualifying"])
    parts += [(q_rows, q_ex), analyse_grid_zero(results), analyse_statuses(results, status_table),
              analyse_populations(results, t["qualifying"], t["drivers"], t["sprints"]),
              analyse_constructors(results), analyse_retiree_ordering(results),
              analyse_indianapolis(t["races"], results)]

    policies = pd.DataFrame([row for rows, _ in parts for row in rows], columns=POLICY_COLUMNS)
    examples_df = pd.concat([ex for _, ex in parts], ignore_index=True).reindex(columns=EXAMPLE_COLUMNS)
    outputs = {"historical_policy_analysis": policies, "historical_policy_examples": examples_df,
               "historical_status_category_proposal": status_table, "historical_qualifying_coverage": coverage}
    for name, df in outputs.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, encoding="utf-8")
    return outputs


if __name__ == "__main__":
    config = load_config()
    out = run(PROJECT_ROOT / config["paths"]["data_interim"], PROJECT_ROOT / config["paths"]["tables"])
    for name, df in out.items():
        print(f"{name}: {len(df)} rows")
