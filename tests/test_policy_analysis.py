"""
Offline tests for src/policy_analysis.py (Step 5.5).
Checks the proposed category rule, the retiree-order check, the
Indianapolis rule, and that the analysis never modifies its input files.
"""

import hashlib

import pandas as pd
import pytest

from src.policy_analysis import (analyse_indianapolis, propose_result_category, retiree_order_checks, run)


@pytest.mark.parametrize("text, status, laps, expected", [
    ("1", "Finished", "57", "finished"),
    ("12", "+1 Lap", "56", "finished"),
    ("20", "Lapped", "55", "finished"),
    ("17", "Engine", "49", "classified_not_running"),       # retired late but classified
    ("R", "Engine", "20", "started_not_classified"),
    ("R", "+1 Lap", "56", "started_not_classified"),         # 2005 Australian GP (button)
    ("W", "Engine", "0", "did_not_start"),                   # same status, did not start
    ("W", "Withdrew", "24", "ambiguous_withdrawn_after_start"),
    ("D", "Disqualified", "0", "disqualified"),
    ("20", "Disqualified", "44", "disqualified"),            # 2024 Belgian GP (russell): numeric code
    ("4", "Withdrew", "74", "ambiguous_classified_withdrawn"),
    ("19", "Not classified", "49", "ambiguous_classified_not_classified"),
    (None, "Finished", "57", "unknown"),
])
def test_propose_result_category(text, status, laps, expected):
    assert propose_result_category(text, status, laps) == expected


def race_rows(race_id, rows):
    season, rnd = race_id.split("_")
    return pd.DataFrame([{"season": season, "round": str(int(rnd)), "race_id": race_id, "race_name": "GP",
                          "position": p, "position_text": t, "laps": l, "driver_id": f"d{p}"}
                         for p, t, l in rows])


def test_retiree_order_checks_counts_inversions():
    results = pd.concat([
        race_rows("1990_01", [("1", "1", "50"), ("2", "R", "40"), ("3", "R", "30")]),   # ordered
        race_rows("1990_02", [("1", "1", "50"), ("2", "R", "30"), ("3", "R", "35")]),   # inversion
        race_rows("2005_01", [("1", "1", "57"), ("2", "R", "56"), ("3", "3", "55")]),   # R above classified
    ])
    checks = retiree_order_checks(results).set_index("race_id")
    assert checks.loc["1990_01", "laps_inversions"] == 0
    assert checks.loc["1990_02", "laps_inversions"] == 1
    assert bool(checks.loc["2005_01", "retired_above_classified"])
    assert not bool(checks.loc["1990_01", "retired_above_classified"])


def test_indianapolis_rule_uses_race_name_not_circuit():
    races = pd.DataFrame({"season": ["1950", "2005"], "round": ["3", "9"], "race_id": ["1950_03", "2005_09"],
                          "race_name": ["Indianapolis 500", "United States Grand Prix"],
                          "circuit_id": ["indianapolis", "indianapolis"]})
    results = pd.DataFrame({"season": ["1950", "2005"], "round": ["3", "9"], "race_id": ["1950_03", "2005_09"],
                            "race_name": ["Indianapolis 500", "United States Grand Prix"],
                            "driver_id": ["parsons", "schumacher"]})
    rows, _ = analyse_indianapolis(races, results)
    assert rows[0]["affected_rows"] == 1 and rows[0]["affected_races"] == 1
    assert "United States Grand Prix" in rows[0]["findings"]
    assert rows[0]["decision_status"] == "PENDING REVIEW"


def test_run_writes_outputs_and_does_not_modify_inputs(tmp_path):
    """Tiny interim set in the real column layout; inputs must be byte-identical afterwards."""
    from src.collect import flatten_qualifying, flatten_results, flatten_schedule

    def race(rnd, **records):
        return {"season": "2005", "round": str(rnd), "raceName": "Australian Grand Prix", "date": "2005-03-06",
                "Circuit": {"circuitId": "albert_park", "circuitName": "Albert Park",
                            "Location": {"lat": "-37.8", "long": "144.9", "locality": "Melbourne",
                                         "country": "Australia"}}, **records}

    def rec(pos, driver, text=None, status="Finished", laps="57", grid="1", number="1"):
        return {"number": number, "position": str(pos), "positionText": text or str(pos), "points": "0",
                "grid": grid, "laps": laps, "status": status,
                "Driver": {"driverId": driver, "givenName": driver, "familyName": "X"},
                "Constructor": {"constructorId": "bar", "name": "BAR"}}

    races = [race(1, Results=[rec(1, "fisichella"), rec(2, "button", "R", "+1 Lap", "56", "0"),
                              rec(3, "sato", "W", "Withdrew", "0", number="4")])]
    quali = [race(1, QualifyingResults=[{"number": "1", "position": "1", "Driver": {"driverId": "fisichella"},
                                         "Constructor": {"constructorId": "bar"}, "Q1": "1:30.0"}])]
    interim = tmp_path / "interim"
    interim.mkdir()
    flatten_schedule(races).to_csv(interim / "races.csv", index=False)
    flatten_results(races).to_csv(interim / "results.csv", index=False)
    flatten_qualifying(quali).to_csv(interim / "qualifying.csv", index=False)
    flatten_results([], "SprintResults").to_csv(interim / "sprints.csv", index=False)
    pd.DataFrame({"season": ["2005"] * 4, "driver_id": ["fisichella", "button", "sato", "listonly"],
                  "driver_first_name": ["a"] * 4, "driver_last_name": ["b"] * 4}).to_csv(interim / "drivers.csv", index=False)
    pd.DataFrame({"season": ["2005"], "constructor_id": ["bar"]}).to_csv(interim / "constructors.csv", index=False)

    digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in interim.glob("*.csv")}
    before = digest()
    out_dir = tmp_path / "tables"
    out_dir.mkdir()
    outputs = run(interim, out_dir)

    assert digest() == before  # read-only
    assert {p.stem for p in out_dir.glob("*.csv")} == {"historical_policy_analysis", "historical_policy_examples",
                                                       "historical_status_category_proposal",
                                                       "historical_qualifying_coverage"}
    policies = outputs["historical_policy_analysis"]
    assert (policies["decision_status"] == "PENDING REVIEW").all()
    assert set(policies["issue_id"].astype(str)) == {str(i) for i in range(1, 9)}
    cats = outputs["historical_status_category_proposal"].set_index("status")["proposed_category"]
    assert cats["+1 Lap"] == "started_not_classified" and cats["Withdrew"] == "did_not_start"
