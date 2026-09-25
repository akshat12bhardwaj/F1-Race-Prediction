"""
Collect one F1 season through the cached Jolpica client and flatten the JSON
into simple tables (one row per race, per driver-race, per driver, ...).

Important: values are kept exactly as the API sends them (text). A field
the API leaves out (e.g. Q3 for a driver knocked out in Q1, FastestLap for a
retirement) becomes None, which is an empty cell in the CSV. No type
conversion, cleaning or feature engineering happens here; that belongs to
the cleaning step (Step 8) and later.

Collection uses season-level endpoints with pagination (limit=100), as in
blueprint section 15: e.g. /2024/results.json pages 0, 100, 200, ...
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

TABLE_NAMES = ["races", "results", "qualifying", "sprints", "drivers", "constructors"]
# Season-level endpoint behind each table (used to read MRData.total for reconciliation)
TABLE_ENDPOINTS = {"races": "races", "results": "results", "qualifying": "qualifying",
                   "sprints": "sprint", "drivers": "drivers", "constructors": "constructors"}

# Column order of each table (also documented in docs/DATA_DICTIONARY.md)
RACE_COLUMNS = ["season", "round", "race_id", "race_name", "race_date", "race_time",
                "circuit_id", "circuit_name", "circuit_lat", "circuit_lon", "locality", "country",
                "sprint_scheduled"]
RACE_KEY_COLUMNS = ["season", "round", "race_id", "race_name", "race_date", "race_time",
                    "circuit_id", "circuit_name", "locality", "country"]
DRIVER_COLUMNS = ["driver_id", "driver_code", "permanent_number", "driver_number",
                  "driver_first_name", "driver_last_name", "driver_dob", "driver_nationality"]
CONSTRUCTOR_COLUMNS = ["constructor_id", "constructor_name", "constructor_nationality"]
RESULT_COLUMNS = (RACE_KEY_COLUMNS + DRIVER_COLUMNS + CONSTRUCTOR_COLUMNS +
                  ["grid", "position", "position_text", "points", "laps", "status",
                   "time_millis", "time_text", "fastest_lap_rank", "fastest_lap_lap",
                   "fastest_lap_time", "fastest_lap_avg_speed", "fastest_lap_speed_units"])
QUALIFYING_COLUMNS = (RACE_KEY_COLUMNS + DRIVER_COLUMNS + CONSTRUCTOR_COLUMNS +
                      ["qualifying_position", "q1", "q2", "q3"])


def make_race_id(season, round_):
    """Blueprint race_id format, e.g. season "2024", round "1" -> "2024_01"."""
    return f"{int(season)}_{int(round_):02d}"


def _race_fields(race):
    circuit = race["Circuit"]
    location = circuit.get("Location", {})
    return {
        "season": race["season"], "round": race["round"],
        "race_id": make_race_id(race["season"], race["round"]),
        "race_name": race.get("raceName"), "race_date": race.get("date"), "race_time": race.get("time"),
        "circuit_id": circuit.get("circuitId"), "circuit_name": circuit.get("circuitName"),
        "circuit_lat": location.get("lat"), "circuit_lon": location.get("long"),
        "locality": location.get("locality"), "country": location.get("country"),
    }


def _driver_fields(record):
    driver = record.get("Driver", {})
    return {
        "driver_id": driver.get("driverId"), "driver_code": driver.get("code"),
        "permanent_number": driver.get("permanentNumber"),  # driver's CURRENT number (Step 2, F-001)
        "driver_number": record.get("number"),                # car number used in THIS race
        "driver_first_name": driver.get("givenName"), "driver_last_name": driver.get("familyName"),
        "driver_dob": driver.get("dateOfBirth"), "driver_nationality": driver.get("nationality"),
    }


def _constructor_fields(record):
    constructor = record.get("Constructor", {})
    return {"constructor_id": constructor.get("constructorId"),
            "constructor_name": constructor.get("name"),
            "constructor_nationality": constructor.get("nationality")}


def flatten_schedule(races):
    """One row per race from the season calendar."""
    rows = []
    for race in races:
        row = _race_fields(race)
        row["sprint_scheduled"] = "Sprint" in race  # calendar lists a Sprint session
        rows.append(row)
    return pd.DataFrame(rows, columns=RACE_COLUMNS)


def flatten_results(races, record_key="Results"):
    """One row per driver per race. Used for race results and sprint results
    (both have the same fields, verified in Step 2)."""
    rows = []
    for race in races:
        race_part = {k: v for k, v in _race_fields(race).items() if k in RACE_KEY_COLUMNS}
        for record in race.get(record_key, []):
            fastest = record.get("FastestLap", {})
            speed = fastest.get("AverageSpeed", {})
            rows.append({
                **race_part, **_driver_fields(record), **_constructor_fields(record),
                "grid": record.get("grid"), "position": record.get("position"),
                "position_text": record.get("positionText"), "points": record.get("points"),
                "laps": record.get("laps"), "status": record.get("status"),
                "time_millis": record.get("Time", {}).get("millis"),
                "time_text": record.get("Time", {}).get("time"),
                "fastest_lap_rank": fastest.get("rank"), "fastest_lap_lap": fastest.get("lap"),
                "fastest_lap_time": fastest.get("Time", {}).get("time"),
                "fastest_lap_avg_speed": speed.get("speed"), "fastest_lap_speed_units": speed.get("units"),
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def flatten_qualifying(races):
    """One row per driver per qualifying session. Missing Q2/Q3 stay None."""
    rows = []
    for race in races:
        race_part = {k: v for k, v in _race_fields(race).items() if k in RACE_KEY_COLUMNS}
        for record in race.get("QualifyingResults", []):
            rows.append({**race_part, **_driver_fields(record), **_constructor_fields(record),
                         "qualifying_position": record.get("position"),
                         "q1": record.get("Q1"), "q2": record.get("Q2"), "q3": record.get("Q3")})
    return pd.DataFrame(rows, columns=QUALIFYING_COLUMNS)


def flatten_drivers(drivers, season):
    """Season driver list (/{season}/drivers.json)."""
    rows = [{"season": str(season), "driver_id": d.get("driverId"), "driver_code": d.get("code"),
             "permanent_number": d.get("permanentNumber"), "driver_first_name": d.get("givenName"),
             "driver_last_name": d.get("familyName"), "driver_dob": d.get("dateOfBirth"),
             "driver_nationality": d.get("nationality")} for d in drivers]
    return pd.DataFrame(rows, columns=["season", "driver_id", "driver_code", "permanent_number",
                                       "driver_first_name", "driver_last_name", "driver_dob",
                                       "driver_nationality"])


def flatten_constructors(constructors, season):
    """Season constructor list (/{season}/constructors.json)."""
    rows = [{"season": str(season), "constructor_id": c.get("constructorId"),
             "constructor_name": c.get("name"), "constructor_nationality": c.get("nationality")}
            for c in constructors]
    return pd.DataFrame(rows, columns=["season"] + CONSTRUCTOR_COLUMNS)


def _season_list(client, season, endpoint, table, item):
    """All items of a non-race list endpoint, e.g. /2024/drivers.json -> DriverTable.Drivers."""
    pages = client.get_paginated(f"/{int(season)}/{endpoint}.json")
    items = [x for page in pages for x in page["MRData"][table][item]]
    if len(items) != int(pages[0]["MRData"]["total"]):
        raise ValueError(f"/{season}/{endpoint}.json: expected {pages[0]['MRData']['total']}, got {len(items)}")
    return items


def collect_season(client, season):
    """
    Fetch one season through the cached client and return raw-value tables:
    races, results, qualifying, sprints, drivers, constructors.
    """
    return {
        "races": flatten_schedule(client.get_schedule(season)),
        "results": flatten_results(client.get_race_results(season), "Results"),
        "qualifying": flatten_qualifying(client.get_qualifying(season)),
        "sprints": flatten_results(client.get_sprint_results(season), "SprintResults"),
        "drivers": flatten_drivers(_season_list(client, season, "drivers", "DriverTable", "Drivers"), season),
        "constructors": flatten_constructors(
            _season_list(client, season, "constructors", "ConstructorTable", "Constructors"), season),
    }


def race_summary(tables):
    """
    One row per scheduled race: record counts per source, duplicate
    (season, round, driver_id) keys in results, and missing IDs in results.
    Duplicates are counted, never dropped.
    """
    races = tables["races"][["season", "round", "race_id", "race_name"]].copy()
    key = ["season", "round"]

    def count(df, name):
        return df.groupby(key).size().rename(name).reset_index()

    results = tables["results"]
    duplicates = (results[results.duplicated(key + ["driver_id"], keep=False)]
                  .groupby(key).size().rename("duplicate_result_keys").reset_index())
    missing_driver = results[results["driver_id"].isna()].groupby(key).size().rename("missing_driver_ids").reset_index()
    missing_constructor = (results[results["constructor_id"].isna()].groupby(key).size()
                           .rename("missing_constructor_ids").reset_index())

    summary = races
    for part in [count(results, "number_of_results"), count(tables["qualifying"], "number_of_qualifying_records"),
                 count(tables["sprints"], "number_of_sprint_records"), duplicates, missing_driver,
                 missing_constructor]:
        summary = summary.merge(part, on=key, how="left")
    count_columns = [c for c in summary.columns if c not in ["season", "round", "race_id", "race_name"]]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    return summary.sort_values("round", key=lambda s: s.astype(int)).reset_index(drop=True)


# ----------------------------------------------------------------------
# Step 5: many seasons
# ----------------------------------------------------------------------
def api_totals(client, season):
    """MRData.total of each season-level endpoint, read from the first cached page (no new request)."""
    return {name: int(client.get_json(f"/{int(season)}/{endpoint}.json",
                                      {"limit": client.page_limit, "offset": 0})["MRData"]["total"])
            for name, endpoint in TABLE_ENDPOINTS.items()}


def collect_history(client, seasons):
    """
    Collect every season in `seasons` (e.g. range(1950, 2026)) with collect_season().

    Returns (tables, api_totals_df):
      tables        : dict of combined DataFrames (seasons in order, API order inside a season)
      api_totals_df : one row per season with MRData.total of every endpoint

    Resumable: every response goes through the disk cache, so after an
    interruption a re-run only downloads what is still missing.
    """
    parts = {name: [] for name in TABLE_NAMES}
    totals = []
    for season in seasons:
        start = time.monotonic()
        before = dict(client.stats)
        season_tables = collect_season(client, season)
        for name in TABLE_NAMES:
            parts[name].append(season_tables[name])
        totals.append({"season": str(season), **{f"api_total_{k}": v for k, v in api_totals(client, season).items()}})
        new_requests = client.stats["network_requests"] - before["network_requests"]
        logger.info("Season %s: races=%d results=%d qualifying=%d sprints=%d | new requests=%d | %.0f s",
                    season, len(season_tables["races"]), len(season_tables["results"]),
                    len(season_tables["qualifying"]), len(season_tables["sprints"]),
                    new_requests, time.monotonic() - start)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    return tables, pd.DataFrame(totals)


def main(argv=None):
    """Command line: python -m src.collect --start 1950 --end 2025"""
    from src.api_client import JolpicaClient
    from src.collection_checks import write_reports
    from src.config import PROJECT_ROOT, load_config

    parser = argparse.ArgumentParser(description="Collect F1 seasons from the Jolpica API (cached).")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output-prefix", default="",
                        help="optional file-name prefix, e.g. 'subset_' for a test run")
    args = parser.parse_args(argv)

    config = load_config()
    interim_dir = PROJECT_ROOT / config["paths"]["data_interim"]
    tables_dir = PROJECT_ROOT / config["paths"]["tables"]
    logs_dir = PROJECT_ROOT / config["paths"]["logs"]
    for folder in (interim_dir, tables_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = logs_dir / f"collection_{args.start}_{args.end}_{stamp}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")])
    logging.getLogger("src.api_client").setLevel(logging.WARNING)  # keep retries/errors, hide per-file messages

    client = JolpicaClient(config=config)
    logger.info("Collecting seasons %d-%d | spacing %.1f s between network requests",
                args.start, args.end, client.min_interval)
    start = time.monotonic()
    tables, totals = collect_history(client, range(args.start, args.end + 1))
    collection_stats = dict(client.stats)

    for name, df in tables.items():
        df.to_csv(interim_dir / f"{args.output_prefix}{name}.csv", index=False, encoding="utf-8")
    write_reports(client, tables, totals, tables_dir, prefix=args.output_prefix)

    runtime = time.monotonic() - start
    logger.info("Collection stats: %s | runtime %.1f min", collection_stats, runtime / 60)
    for name, df in tables.items():
        logger.info("  %-13s %6d rows", name, len(df))
    run_log = logs_dir / "collection_runs.csv"
    pd.DataFrame([{"finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "seasons": f"{args.start}-{args.end}", "output_prefix": args.output_prefix,
                   **{f"stat_{k}": v for k, v in collection_stats.items()},
                   "runtime_seconds": round(runtime, 1), **{f"rows_{k}": len(v) for k, v in tables.items()}}]
                 ).to_csv(run_log, mode="a", header=not run_log.exists(), index=False)
    return tables


if __name__ == "__main__":
    main()
