"""
Offline tests for src/collect.py (flattening and race summary).
Uses tiny hand-made race dicts in the Jolpica format; no network.
"""

import pandas as pd

from src.collect import (QUALIFYING_COLUMNS, RESULT_COLUMNS, flatten_qualifying, flatten_results,
                         flatten_schedule, make_race_id, race_summary)


def race(round_="1", **records):
    return {"season": "2024", "round": round_, "raceName": f"GP {round_}", "date": "2024-03-02",
            "time": "15:00:00Z",
            "Circuit": {"circuitId": "bahrain", "circuitName": "Bahrain International Circuit",
                        "Location": {"lat": "26.0325", "long": "50.5106", "locality": "Sakhir",
                                     "country": "Bahrain"}},
            **records}


def result(position, driver_id, position_text=None, status="Finished", with_times=True):
    rec = {"number": "4", "position": str(position), "positionText": position_text or str(position),
           "points": "0", "grid": "0", "laps": "57", "status": status,
           "Driver": {"driverId": driver_id, "permanentNumber": "1", "code": "NOR", "givenName": "Lando",
                      "familyName": "Norris", "dateOfBirth": "1999-11-13", "nationality": "British"},
           "Constructor": {"constructorId": "mclaren", "name": "McLaren", "nationality": "British"}}
    if with_times:
        rec["Time"] = {"millis": "5504742", "time": "1:31:44.742"}
        rec["FastestLap"] = {"rank": "1", "lap": "39", "Time": {"time": "1:32.608"},
                             "AverageSpeed": {"units": "kph", "speed": "210.383"}}
    return rec


def test_race_id_format():
    assert make_race_id("2024", "1") == "2024_01"
    assert make_race_id(1950, 7) == "1950_07"


def test_schedule_flags_sprint_weekends():
    races = [race("1"), race("5", Sprint={"date": "2024-04-20"})]
    df = flatten_schedule(races)
    assert df["sprint_scheduled"].tolist() == [False, True]
    assert df.loc[0, "race_id"] == "2024_01"
    assert df.loc[0, "circuit_lat"] == "26.0325"


def test_results_keep_raw_string_values():
    df = flatten_results([race(Results=[result(1, "norris")])])
    assert list(df.columns) == RESULT_COLUMNS
    row = df.iloc[0]
    assert row["position"] == "1" and row["grid"] == "0" and row["points"] == "0"  # text, not numbers
    assert row["permanent_number"] == "1"   # Driver.permanentNumber
    assert row["driver_number"] == "4"      # car number of this race
    assert row["fastest_lap_avg_speed"] == "210.383"


def test_missing_fields_stay_missing():
    df = flatten_results([race(Results=[result(20, "zhou", "R", "Retired", with_times=False)])])
    row = df.iloc[0]
    assert row["position_text"] == "R" and row["status"] == "Retired"
    assert row["time_millis"] is None and row["fastest_lap_rank"] is None


def test_sprint_uses_same_columns():
    df = flatten_results([race("5", SprintResults=[result(1, "verstappen")])], "SprintResults")
    assert list(df.columns) == RESULT_COLUMNS and len(df) == 1


def test_qualifying_missing_q2_q3_not_invented():
    quali = {"number": "2", "position": "18", "Q1": "1:30.770",
             "Driver": {"driverId": "sargeant"}, "Constructor": {"constructorId": "williams"}}
    df = flatten_qualifying([race(QualifyingResults=[quali])])
    assert list(df.columns) == QUALIFYING_COLUMNS
    assert df.loc[0, "q1"] == "1:30.770"
    assert df.loc[0, "q2"] is None and df.loc[0, "q3"] is None


def test_race_summary_counts_and_reports_duplicates():
    races = [race("1"), race("2")]
    results = flatten_results([race("1", Results=[result(1, "a"), result(2, "a")]),   # duplicate driver
                               race("2", Results=[result(1, "b")])])
    tables = {"races": flatten_schedule(races), "results": results,
              "qualifying": flatten_qualifying([]), "sprints": flatten_results([], "SprintResults")}
    summary = race_summary(tables)
    assert summary["number_of_results"].tolist() == [2, 1]
    assert summary["duplicate_result_keys"].tolist() == [2, 0]   # both rows of the duplicate are reported
    assert summary["number_of_sprint_records"].tolist() == [0, 0]
    assert len(results) == 3                                      # nothing was dropped


def test_race_summary_reports_missing_ids():
    rec = result(1, "a")
    rec["Driver"] = {}
    tables = {"races": flatten_schedule([race("1")]),
              "results": flatten_results([race("1", Results=[rec])]),
              "qualifying": flatten_qualifying([]), "sprints": flatten_results([], "SprintResults")}
    summary = race_summary(tables)
    assert summary.loc[0, "missing_driver_ids"] == 1
    assert summary.loc[0, "missing_constructor_ids"] == 0
    assert isinstance(tables["results"], pd.DataFrame)
