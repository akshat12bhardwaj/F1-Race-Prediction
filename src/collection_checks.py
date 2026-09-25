"""
Validation and edge-case reports for the collected historical tables (Step 5).

Nothing here changes, drops or re-codes data. Every function only COUNTS and
FLAGS, so later steps (validation, cleaning) can decide what to do.
Numeric checks use temporary pd.to_numeric copies; stored tables stay text.
"""

import pandas as pd

KEY = ["season", "round", "driver_id"]
ID_COLUMNS = ["race_id", "driver_id", "constructor_id"]
NON_STARTER_STATUSES = {"Did not start", "Withdrew", "Did not qualify", "Did not prequalify"}
FINISHED_STATUS = r"^(Finished|Lapped|\+\d+ Laps?)$"

# Fields seen in the Step 2 exploration. Anything else is reported as "unexpected" (not an error).
KNOWN_FIELDS = {
    "results": {"number", "position", "positionText", "points", "Driver", "Constructor",
                "grid", "laps", "status", "Time", "FastestLap"},
    "qualifying": {"number", "position", "Driver", "Constructor", "Q1", "Q2", "Q3"},
    "Driver": {"driverId", "permanentNumber", "code", "url", "givenName", "familyName",
               "dateOfBirth", "nationality"},
    "Constructor": {"constructorId", "url", "name", "nationality"},
}
KNOWN_FIELDS["sprint"] = KNOWN_FIELDS["results"]


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _season_order(values):
    return sorted(set(values), key=int)


# ----------------------------------------------------------------------
# Season level
# ----------------------------------------------------------------------
def season_summary(tables, api_totals):
    """One row per season: counts, uniques, duplicates, missing IDs, and API totals for comparison."""
    races, results, quali, sprints = (tables[k] for k in ["races", "results", "qualifying", "sprints"])
    rows = []
    for season in _season_order(races["season"]):
        r, q, s = (df[df["season"] == season] for df in (results, quali, sprints))
        ra = races[races["season"] == season]
        all_rows = pd.concat([r[ID_COLUMNS], q[ID_COLUMNS], s[ID_COLUMNS]])
        rows.append({
            "season": season,
            "races": len(ra),
            "races_missing_race_time": int(ra["race_time"].isna().sum()),
            "result_rows": len(r), "qualifying_rows": len(q), "sprint_rows": len(s),
            "unique_drivers": r["driver_id"].nunique(),
            "unique_constructors": r["constructor_id"].nunique(),
            "drivers_in_season_list": int((tables["drivers"]["season"] == season).sum()),
            "constructors_in_season_list": int((tables["constructors"]["season"] == season).sum()),
            "duplicate_result_keys": int(r.duplicated(KEY, keep=False).sum()),
            "duplicate_qualifying_keys": int(q.duplicated(KEY, keep=False).sum()),
            "duplicate_sprint_keys": int(s.duplicated(KEY, keep=False).sum()),
            "missing_race_ids": int(all_rows["race_id"].isna().sum()),
            "missing_driver_ids": int(all_rows["driver_id"].isna().sum()),
            "missing_constructor_ids": int(all_rows["constructor_id"].isna().sum()),
        })
    summary = pd.DataFrame(rows).merge(api_totals, on="season", how="left")
    # Our row counts must equal what the API reported (MRData.total)
    summary["matches_api_totals"] = (
        (summary["races"] == summary["api_total_races"]) &
        (summary["result_rows"] == summary["api_total_results"]) &
        (summary["qualifying_rows"] == summary["api_total_qualifying"]) &
        (summary["sprint_rows"] == summary["api_total_sprints"]) &
        (summary["drivers_in_season_list"] == summary["api_total_drivers"]) &
        (summary["constructors_in_season_list"] == summary["api_total_constructors"]))
    return summary


# ----------------------------------------------------------------------
# Race level
# ----------------------------------------------------------------------
def race_validation(tables):
    """One row per race with counts and a semicolon-separated list of flags (nothing is removed)."""
    races = tables["races"][["season", "round", "race_id", "race_name", "race_date", "race_time",
                             "sprint_scheduled"]].copy()
    results = tables["results"]
    key = ["season", "round"]

    def per_race(df, name, mask=None):
        df = df if mask is None else df[mask]
        return df.groupby(key).size().rename(name)

    parts = [
        per_race(results, "result_count"),
        per_race(tables["qualifying"], "qualifying_count"),
        per_race(tables["sprints"], "sprint_count"),
        per_race(results, "duplicate_keys", results.duplicated(KEY, keep=False)),
        per_race(results, "missing_ids", results[ID_COLUMNS].isna().any(axis=1)),
        per_race(results, "grid_zero", _num(results["grid"]) == 0),
        # rows sharing a position with another row of the same race (e.g. 1950s shared cars)
        per_race(results, "repeated_position_rows", results.duplicated(["season", "round", "position"], keep=False)),
        per_race(results, "non_numeric_position_text", ~results["position_text"].fillna("").str.isdigit()),
    ]
    out = races.join(pd.concat(parts, axis=1), on=key)
    count_cols = [p.name for p in parts]
    out[count_cols] = out[count_cols].fillna(0).astype(int)

    out["season_median_result_count"] = out.groupby("season")["result_count"].transform("median")
    season_has_quali = out.groupby("season")["qualifying_count"].transform("sum") > 0

    def flags(row, has_quali):
        f = []
        if row["result_count"] == 0: f.append("no_results")
        if row["result_count"] != row["season_median_result_count"]: f.append("result_count_differs_from_season_median")
        if has_quali and row["qualifying_count"] == 0: f.append("no_qualifying_in_qualifying_season")
        if row["qualifying_count"] > 0 and row["qualifying_count"] != row["result_count"]: f.append("qualifying_count_differs_from_results")
        if row["sprint_scheduled"] and row["sprint_count"] == 0: f.append("sprint_scheduled_but_no_sprint_results")
        if not row["sprint_scheduled"] and row["sprint_count"] > 0: f.append("sprint_results_but_not_scheduled")
        if pd.isna(row["race_time"]): f.append("missing_race_time")
        if row["duplicate_keys"]: f.append("duplicate_keys")
        if row["missing_ids"]: f.append("missing_ids")
        if row["grid_zero"]: f.append("grid_zero")
        if row["repeated_position_rows"]: f.append("repeated_position")
        if "Indianapolis 500" in str(row["race_name"]): f.append("indianapolis_500")
        return ";".join(f)

    out["flags"] = [flags(row, hq) for (_, row), hq in zip(out.iterrows(), season_has_quali)]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# Status vocabulary and field coverage
# ----------------------------------------------------------------------
def status_vocabulary(tables):
    """Every (source, status, positionText type) combination with counts and first/last season."""
    both = pd.concat([tables["results"].assign(source="race"), tables["sprints"].assign(source="sprint")],
                     ignore_index=True)
    text = both["position_text"].fillna("<missing>")
    both["position_text_type"] = text.where(~text.str.isdigit(), "numeric")
    both["season_int"] = both["season"].astype(int)
    return (both.groupby(["source", "status", "position_text_type"], dropna=False)
            .agg(records=("season", "size"), first_season=("season_int", "min"), last_season=("season_int", "max"))
            .reset_index().sort_values(["source", "records"], ascending=[True, False]).reset_index(drop=True))


def field_coverage(client, seasons):
    """
    For each season and endpoint, how many records contain each JSON field.
    Reads the raw cached responses again (cache hits, no new downloads), so
    fields that our flat tables do not keep are still inspected.
    """
    fetch = {"results": ("get_race_results", "Results"), "qualifying": ("get_qualifying", "QualifyingResults"),
             "sprint": ("get_sprint_results", "SprintResults")}
    rows = []
    for season in seasons:
        for endpoint, (method, record_key) in fetch.items():
            races = getattr(client, method)(int(season))
            records = [rec for race in races for rec in race.get(record_key, [])]
            if not records:
                continue
            for level, getter in [(endpoint, lambda r: r), ("Driver", lambda r: r.get("Driver", {})),
                                  ("Constructor", lambda r: r.get("Constructor", {}))]:
                counts = pd.Series([k for rec in records for k in getter(rec).keys()]).value_counts()
                for field, n in counts.items():
                    rows.append({"endpoint": endpoint, "season": str(season), "level": level, "field": field,
                                 "records": len(records), "records_with_field": int(n),
                                 "share": round(n / len(records), 4),
                                 "known_field": field in KNOWN_FIELDS[level]})
    return pd.DataFrame(rows, columns=["endpoint", "season", "level", "field", "records",
                                       "records_with_field", "share", "known_field"])


# ----------------------------------------------------------------------
# Edge cases (long format: one row per observation)
# ----------------------------------------------------------------------
EDGE_COLUMNS = ["category", "season", "round", "race_id", "driver_id", "constructor_id", "detail"]


def _rows(df, category, detail):
    """Turn selected result rows into edge-case rows; detail(row) builds the text."""
    return [{"category": category, "season": r["season"], "round": r["round"], "race_id": r["race_id"],
             "driver_id": r["driver_id"], "constructor_id": r["constructor_id"], "detail": detail(r)}
            for _, r in df.iterrows()]


def edge_cases(tables, race_checks, coverage):
    """Collect unusual observations into one table. Categories are described in DATA_DICTIONARY.md."""
    results, quali, sprints = tables["results"], tables["qualifying"], tables["sprints"]
    races, drivers, constructors = tables["races"], tables["drivers"], tables["constructors"]
    out = []

    def race_row(category, r, detail):
        out.append({"category": category, "season": r["season"], "round": r["round"], "race_id": r["race_id"],
                    "driver_id": None, "constructor_id": None, "detail": detail})

    # Race counts
    for _, r in race_checks[race_checks["flags"].str.contains("result_count_differs")].iterrows():
        race_row("result_count_differs_from_season_median", r,
                 f"{r['result_count']} results vs season median {r['season_median_result_count']:g}")
    for _, r in race_checks[race_checks["flags"].str.contains("indianapolis_500")].iterrows():
        race_row("indianapolis_500", r, f"{r['race_name']}: {r['result_count']} results")
    for _, r in race_checks[race_checks["flags"].str.contains("no_qualifying_in_qualifying_season")].iterrows():
        race_row("race_without_qualifying", r, "season has qualifying data but this race has none")
    for _, r in race_checks[race_checks["flags"].str.contains("qualifying_count_differs")].iterrows():
        race_row("qualifying_count_differs_from_results", r,
                 f"qualifying {r['qualifying_count']} vs results {r['result_count']}")
    for flag in ["sprint_scheduled_but_no_sprint_results", "sprint_results_but_not_scheduled"]:
        for _, r in race_checks[race_checks["flags"].str.contains(flag)].iterrows():
            race_row(flag, r, flag.replace("_", " "))

    # Season level: qualifying availability, missing race times, retirements
    for season in _season_order(races["season"]):
        rc = race_checks[race_checks["season"] == season]
        base = {"season": season, "round": None, "race_id": None, "driver_id": None, "constructor_id": None}
        if rc["qualifying_count"].sum() == 0:
            out.append({**base, "category": "season_without_qualifying", "detail": "0 qualifying records"})
        missing_time = int(rc["race_time"].isna().sum())
        if missing_time:
            out.append({**base, "category": "missing_race_time", "detail": f"{missing_time} of {len(rc)} races"})
        r = results[results["season"] == season]
        text = r["position_text"].fillna("")
        for code, n in text[~text.str.isdigit()].value_counts().items():
            out.append({**base, "category": "position_text_code", "detail": f"code={code!r} records={n}"})
        classified_unfinished = r[text.str.isdigit() & ~r["status"].fillna("").str.match(FINISHED_STATUS)]
        if len(classified_unfinished):
            out.append({**base, "category": "classified_but_not_finished",
                        "detail": f"{len(classified_unfinished)} records with numeric positionText and status "
                                  f"not Finished/Lapped/+N Laps (e.g. {classified_unfinished['status'].iloc[0]!r})"})

    # Individual records
    out += _rows(results[_num(results["grid"]) == 0], "grid_zero",
                 lambda r: f"grid 0, position {r['position']} ({r['position_text']}), status {r['status']}, laps {r['laps']}")
    non_starter = results["status"].isin(NON_STARTER_STATUSES) | results["position_text"].isin(["W", "F"])
    out += _rows(results[non_starter], "non_starter",
                 lambda r: f"status {r['status']!r}, positionText {r['position_text']!r}, grid {r['grid']}, laps {r['laps']}")
    dsq = results["status"].fillna("").str.contains("isqualif|xclu", case=False) | results["position_text"].isin(["D", "E"])
    out += _rows(results[dsq], "disqualified_or_excluded",
                 lambda r: f"status {r['status']!r}, positionText {r['position_text']!r}, position {r['position']}")
    dup = results[results.duplicated(KEY, keep=False)]
    out += _rows(dup, "duplicate_result_key",
                 lambda r: f"car {r['driver_number']}, position {r['position']} ({r['position_text']}), "
                           f"status {r['status']}, points {r['points']}")
    shared = results[results.duplicated(["season", "round", "position"], keep=False)]
    out += _rows(shared, "repeated_position",
                 lambda r: f"position {r['position']} ({r['position_text']}) shared in this race, car {r['driver_number']}, "
                           f"status {r['status']}, points {r['points']}")
    out += _rows(quali[quali.duplicated(KEY, keep=False)], "duplicate_qualifying_key",
                 lambda r: f"car {r['driver_number']}, qualifying position {r['qualifying_position']}")
    out += _rows(sprints[sprints.duplicated(KEY, keep=False)], "duplicate_sprint_key",
                 lambda r: f"car {r['driver_number']}, position {r['position']}, status {r['status']}")

    # Drivers absent from a race between two appearances in adjacent rounds of the same season
    appeared = set(zip(results["season"], results["round"].astype(int), results["driver_id"]))
    rounds_per_season = races.groupby("season")["round"].apply(lambda s: set(s.astype(int))).to_dict()
    for season, rnd, driver in sorted(appeared, key=lambda x: (int(x[0]), x[1], x[2])):
        nxt = rnd + 2
        if (season, nxt, driver) in appeared and (season, rnd + 1, driver) not in appeared \
                and rnd + 1 in rounds_per_season.get(season, set()):
            race = races[(races["season"] == season) & (races["round"].astype(int) == rnd + 1)].iloc[0]
            out.append({"category": "absent_between_appearances", "season": season, "round": str(rnd + 1),
                        "race_id": race["race_id"], "driver_id": driver, "constructor_id": None,
                        "detail": f"raced in rounds {rnd} and {nxt} but has no record in round {rnd + 1}"})

    # Qualifying vs results driver sets (only races that have qualifying)
    q_sets = quali.groupby(["season", "round"])["driver_id"].apply(set)
    r_sets = results.groupby(["season", "round"])["driver_id"].apply(set)
    for (season, rnd), q_ids in q_sets.items():
        r_ids = r_sets.get((season, rnd), set())
        race_id = f"{int(season)}_{int(rnd):02d}"
        for d in sorted(q_ids - r_ids):
            out.append({"category": "in_qualifying_not_in_results", "season": season, "round": rnd,
                        "race_id": race_id, "driver_id": d, "constructor_id": None, "detail": ""})
        for d in sorted(r_ids - q_ids):
            out.append({"category": "in_results_not_in_qualifying", "season": season, "round": rnd,
                        "race_id": race_id, "driver_id": d, "constructor_id": None, "detail": ""})

    # Season lists vs records
    raced = set(zip(results["season"], results["driver_id"])) | set(zip(quali["season"], quali["driver_id"])) \
        | set(zip(sprints["season"], sprints["driver_id"]))
    for _, d in drivers.iterrows():
        if (d["season"], d["driver_id"]) not in raced:
            out.append({"category": "season_list_driver_without_records", "season": d["season"], "round": None,
                        "race_id": None, "driver_id": d["driver_id"], "constructor_id": None,
                        "detail": f"{d['driver_first_name']} {d['driver_last_name']}"})
    listed = set(zip(drivers["season"], drivers["driver_id"]))
    for season, driver in sorted({x for x in zip(results["season"], results["driver_id"]) if x not in listed}):
        out.append({"category": "result_driver_not_in_season_list", "season": season, "round": None,
                    "race_id": None, "driver_id": driver, "constructor_id": None, "detail": ""})

    # Identity changes over time
    cons = results[["constructor_id", "constructor_name", "season"]].drop_duplicates()
    for cid, g in cons.groupby("constructor_id"):
        names = g.groupby("constructor_name")["season"].agg(lambda s: f"{min(s, key=int)}-{max(s, key=int)}")
        if len(names) > 1:
            out.append({"category": "constructor_id_with_multiple_names", "season": None, "round": None,
                        "race_id": None, "driver_id": None, "constructor_id": cid,
                        "detail": "; ".join(f"{n} ({span})" for n, span in names.items())})
    for name, g in cons.groupby("constructor_name"):
        if g["constructor_id"].nunique() > 1:
            out.append({"category": "constructor_name_with_multiple_ids", "season": None, "round": None,
                        "race_id": None, "driver_id": None, "constructor_id": None,
                        "detail": f"{name}: {sorted(g['constructor_id'].unique())}"})
    people = results[["driver_id", "driver_first_name", "driver_last_name", "driver_dob"]].drop_duplicates()
    for did, g in people.groupby("driver_id"):
        if len(g) > 1:
            out.append({"category": "driver_id_with_multiple_identities", "season": None, "round": None,
                        "race_id": None, "driver_id": did, "constructor_id": None,
                        "detail": "; ".join(f"{a} {b} ({c})" for a, b, c in g.iloc[:, 1:].itertuples(index=False))})
    for person, g in people.groupby(["driver_first_name", "driver_last_name", "driver_dob"], dropna=False):
        if g["driver_id"].nunique() > 1:
            out.append({"category": "same_person_multiple_driver_ids", "season": None, "round": None,
                        "race_id": None, "driver_id": None, "constructor_id": None,
                        "detail": f"{person}: {sorted(g['driver_id'].unique())}"})

    # Unexpected JSON fields
    for _, c in coverage[~coverage["known_field"]].iterrows():
        out.append({"category": "unexpected_field", "season": c["season"], "round": None, "race_id": None,
                    "driver_id": None, "constructor_id": None,
                    "detail": f"{c['endpoint']}/{c['level']}.{c['field']} in {c['records_with_field']} of {c['records']} records"})

    return pd.DataFrame(out, columns=EDGE_COLUMNS)


def write_reports(client, tables, api_totals, tables_dir, prefix=""):
    """Build every Step 5 report and save it to outputs/tables/. Returns the reports."""
    seasons = _season_order(tables["races"]["season"])
    race_checks = race_validation(tables)
    coverage = field_coverage(client, seasons)
    reports = {
        "historical_collection_summary": season_summary(tables, api_totals),
        "historical_race_validation": race_checks,
        "historical_status_vocabulary": status_vocabulary(tables),
        "historical_field_coverage": coverage,
        "historical_edge_cases": edge_cases(tables, race_checks, coverage),
    }
    for name, df in reports.items():
        df.to_csv(tables_dir / f"{prefix}{name}.csv", index=False, encoding="utf-8")
    return reports
