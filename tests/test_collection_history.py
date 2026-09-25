"""
Offline tests for multi-season collection (src/collect.py: collect_history)
and the Step 5 reports (src/collection_checks.py).

A FakeClient returns hand-made Jolpica-style data for two seasons:
  1955: no qualifying, no sprint, no race start times; a shared drive
        (same driver twice in one race), grid 0, a non-starter, a
        disqualification, a renamed constructor, a list-only driver
  2021: qualifying + sprint + start times; one driver absent in round 2
"""

import pandas as pd
import pytest

from src.collect import collect_history
from src.collection_checks import (edge_cases, field_coverage, race_validation, season_summary,
                                   status_vocabulary)


def rec(pos, driver, team="ferrari", team_name="Ferrari", text=None, status="Finished", grid="1",
        number="1", extra=None):
    r = {"number": number, "position": str(pos), "positionText": text or str(pos), "points": "0",
         "grid": grid, "laps": "50", "status": status,
         "Driver": {"driverId": driver, "givenName": driver.title(), "familyName": "X",
                    "dateOfBirth": "1920-01-01", "nationality": "Italian"},
         "Constructor": {"constructorId": team, "name": team_name, "nationality": "Italian"}}
    r.update(extra or {})
    return r


def race(season, rnd, time=None, sprint=False, **records):
    r = {"season": season, "round": str(rnd), "raceName": f"GP {rnd}", "date": f"{season}-06-0{rnd}",
         "Circuit": {"circuitId": "monza", "circuitName": "Monza",
                     "Location": {"lat": "45.6", "long": "9.28", "locality": "Monza", "country": "Italy"}},
         **records}
    if time:
        r["time"] = time
    if sprint:
        r["Sprint"] = {"date": f"{season}-06-0{rnd}"}
    return r


RESULTS = {
    1955: [race("1955", 1, Results=[rec(1, "fangio", number="2"), rec(2, "ascari", grid="0"),
                                    rec(3, "fangio", number="4", text="3"),       # shared drive: fangio twice
                                    rec(4, "moss", text="W", status="Did not start")]),
           race("1955", 2, Results=[rec(1, "ascari", team="lancia", team_name="Lancia"),
                                    rec(2, "moss", text="D", status="Disqualified", extra={"OddField": "x"}),
                                    rec(3, "fangio", team="ferrari", team_name="Scuderia Ferrari")])],
    2021: [race("2021", 1, time="13:00:00Z", Results=[rec(1, "hamilton"), rec(2, "verstappen")]),
           race("2021", 2, time="13:00:00Z", sprint=True, Results=[rec(1, "verstappen")]),   # hamilton absent
           race("2021", 3, time="13:00:00Z", Results=[rec(1, "hamilton"), rec(2, "verstappen")])],
}
QUALI = {1955: [], 2021: [race("2021", r, QualifyingResults=[
    {"number": "44", "position": "1", "Driver": {"driverId": "hamilton"}, "Constructor": {"constructorId": "ferrari"},
     "Q1": "1:30.0"},
    {"number": "33", "position": "2", "Driver": {"driverId": "verstappen"}, "Constructor": {"constructorId": "ferrari"},
     "Q1": "1:30.1"}]) for r in (1, 3)]}
SPRINT = {1955: [], 2021: [race("2021", 2, SprintResults=[rec(1, "verstappen"), rec(2, "hamilton")])]}
DRIVERS = {1955: ["fangio", "ascari", "moss", "listonly"], 2021: ["hamilton", "verstappen"]}
CONSTRUCTORS = {1955: ["ferrari", "lancia"], 2021: ["ferrari"]}


class FakeClient:
    page_limit = 100

    def __init__(self):
        self.stats = {"cache_hits": 0, "network_requests": 0, "retries": 0}

    def get_schedule(self, season):
        return [{k: v for k, v in r.items() if k not in ("Results",)} for r in RESULTS[season]]

    def get_race_results(self, season):
        return RESULTS[season]

    def get_qualifying(self, season):
        return QUALI[season]

    def get_sprint_results(self, season):
        return SPRINT[season]

    def get_paginated(self, path):
        season, endpoint = int(path.split("/")[1]), path.split("/")[2].replace(".json", "")
        if endpoint == "drivers":
            items = [{"driverId": d, "givenName": d.title(), "familyName": "X"} for d in DRIVERS[season]]
            return [{"MRData": {"total": str(len(items)), "DriverTable": {"Drivers": items}}}]
        items = [{"constructorId": c, "name": c.title()} for c in CONSTRUCTORS[season]]
        return [{"MRData": {"total": str(len(items)), "ConstructorTable": {"Constructors": items}}}]

    def get_json(self, path, params):
        season, endpoint = int(path.split("/")[1]), path.split("/")[2].replace(".json", "")
        count = {"races": len(RESULTS[season]),
                 "results": sum(len(r["Results"]) for r in RESULTS[season]),
                 "qualifying": sum(len(r["QualifyingResults"]) for r in QUALI[season]),
                 "sprint": sum(len(r["SprintResults"]) for r in SPRINT[season]),
                 "drivers": len(DRIVERS[season]), "constructors": len(CONSTRUCTORS[season])}[endpoint]
        return {"MRData": {"total": str(count)}}


@pytest.fixture(scope="module")
def collected():
    client = FakeClient()
    tables, totals = collect_history(client, [1955, 2021])
    checks = race_validation(tables)
    coverage = field_coverage(client, ["1955", "2021"])
    return client, tables, totals, checks, coverage, edge_cases(tables, checks, coverage)


def test_collect_history_combines_seasons_in_order(collected):
    _, tables, totals, *_ = collected
    assert tables["races"]["season"].tolist() == ["1955", "1955", "2021", "2021", "2021"]
    assert len(tables["results"]) == 12
    assert len(tables["qualifying"]) == 4 and len(tables["sprints"]) == 2
    assert tables["results"]["position"].map(type).eq(str).all()     # still raw text
    assert totals["api_total_results"].tolist() == [7, 5]


def test_season_without_qualifying_sprint_or_race_time_is_tolerated(collected):
    _, tables, *_ = collected
    races_1955 = tables["races"][tables["races"]["season"] == "1955"]
    assert races_1955["race_time"].isna().all()
    assert not races_1955["sprint_scheduled"].any()
    assert (tables["qualifying"]["season"] == "1955").sum() == 0


def test_season_summary_matches_api_totals_and_reports_duplicates(collected):
    _, tables, totals, *_ = collected
    summary = season_summary(tables, totals).set_index("season")
    assert summary["matches_api_totals"].all()
    assert summary.loc["1955", "duplicate_result_keys"] == 2         # both fangio rows reported
    assert summary.loc["1955", "races_missing_race_time"] == 2
    assert summary.loc["2021", "sprint_rows"] == 2
    assert len(tables["results"]) == 12                              # nothing dropped


def test_race_validation_flags(collected):
    *_, checks, _, _ = collected
    flags = dict(zip(checks["race_id"], checks["flags"]))
    assert "duplicate_keys" in flags["1955_01"] and "grid_zero" in flags["1955_01"]
    assert "missing_race_time" in flags["1955_01"]
    assert "missing_race_time" not in flags["2021_01"]
    assert "no_qualifying_in_qualifying_season" in flags["2021_02"]
    assert "sprint_scheduled_but_no_sprint_results" not in flags["2021_02"]


def test_edge_cases_cover_historical_situations(collected):
    *_, edges = collected
    cats = edges.groupby("category").size().to_dict()
    assert cats["duplicate_result_key"] == 2
    assert cats["grid_zero"] == 1
    assert cats["non_starter"] == 1
    assert cats["disqualified_or_excluded"] == 1
    assert cats["season_without_qualifying"] == 1
    assert cats["missing_race_time"] == 1
    absent = edges[edges["category"] == "absent_between_appearances"]
    assert absent[["race_id", "driver_id"]].values.tolist() == [["2021_02", "hamilton"]]
    assert "listonly" in edges.loc[edges["category"] == "season_list_driver_without_records", "driver_id"].tolist()
    renamed = edges[edges["category"] == "constructor_id_with_multiple_names"]
    assert renamed["constructor_id"].tolist() == ["ferrari"]
    assert edges[edges["category"] == "unexpected_field"]["detail"].str.contains("OddField").any()


def test_shared_position_is_flagged_not_dropped():
    # 1951-style shared win: two drivers credited with position 1 in the same car
    shared = [race("1951", 1, Results=[rec(1, "fangio", number="8"), rec(1, "fagioli", number="8"),
                                       rec(2, "ascari")])]
    from src.collect import flatten_qualifying, flatten_results, flatten_schedule
    tables = {"races": flatten_schedule(shared), "results": flatten_results(shared),
              "qualifying": flatten_qualifying([]), "sprints": flatten_results([], "SprintResults"),
              "drivers": pd.DataFrame(columns=["season", "driver_id", "driver_first_name", "driver_last_name"]),
              "constructors": pd.DataFrame(columns=["season", "constructor_id"])}
    checks = race_validation(tables)
    assert checks.loc[0, "repeated_position_rows"] == 2
    assert "repeated_position" in checks.loc[0, "flags"]
    edges = edge_cases(tables, checks, pd.DataFrame(columns=["known_field"]))
    assert sorted(edges.loc[edges["category"] == "repeated_position", "driver_id"]) == ["fagioli", "fangio"]
    assert len(tables["results"]) == 3


def test_status_vocabulary_counts_codes(collected):
    _, tables, *_ = collected
    vocab = status_vocabulary(tables)
    row = vocab[(vocab["status"] == "Did not start")].iloc[0]
    assert row["position_text_type"] == "W" and row["records"] == 1 and row["first_season"] == 1955
    assert isinstance(vocab, pd.DataFrame)
