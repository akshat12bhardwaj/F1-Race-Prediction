"""
Tests for src/validate_raw.py (Step 7).

- Unit tests of the check logic and the missing-value classes (tiny frames).
- Mutation tests: the REAL interim data is loaded into memory, deliberately
  broken, and the validator must report FAIL. Nothing is written back.
- Full-run test: no FAIL on the real data, inputs unchanged, identical outputs twice.
Tests that need data/interim are skipped on a fresh clone.
"""

import hashlib

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.validate_raw import (TABLES, Report, cross_tables, load, missing_classes, run, validate_races,
                              validate_results)

INTERIM = PROJECT_ROOT / "data" / "interim"
HAVE_DATA = all((INTERIM / f"{n}.csv").exists() for n in TABLES)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="interim data not built")


# ---------------- unit tests ----------------
def test_report_pass_fail_info():
    rep = Report()
    assert rep.check("a", "equal", 5, 5)
    assert not rep.check("a", "different", 4, 5)
    rep.check("a", "info only", 7)
    assert [c["status"] for c in rep.checks] == ["PASS", "FAIL", "INFO"]


def test_missing_classes_rules():
    races = pd.DataFrame({"season": ["2004", "2010"], "race_time": [None, None]})
    st, hi, le, un, _, _ = missing_classes("races", "race_time", races)
    assert (hi, un) == (1, 1)                       # 2004: historical; 2010: unexpected

    q = pd.DataFrame({"season": ["2004", "2008", "2008"], "q2": [None, None, "1:20.000"]})
    st, hi, le, un, _, _ = missing_classes("qualifying", "q2", q)
    assert (hi, le, un) == (1, 1, 0)                # no Q2 session in 2004; knocked out in 2008

    w = pd.DataFrame({"season": ["1955", "2020"], "weather_status": ["excluded_indianapolis_500", "ok"],
                      "wx_race_temp_mean_c": [None, None]})
    st, hi, le, un, _, _ = missing_classes("weather_race", "wx_race_temp_mean_c", w)
    assert (st, un) == (1, 1)                       # excluded = structural; missing for an ok race = unexpected

    res = pd.DataFrame({"season": ["2010"] * 4, "position_text": ["1", "2", "3", "R"],
                        "status": ["Finished", "Finished", "+1 Lap", "Engine"], "laps": ["50", "50", "49", "10"],
                        "time_millis": [None, "5400000", None, None]})
    st, hi, le, un, _, mask = missing_classes("results", "time_millis", res)
    assert (le, un) == (2, 1) and mask.tolist() == [True, False, False, False]   # winner without time is unexpected

    whole = res.assign(time_millis=None)            # field absent for the whole season -> historically unavailable
    st, hi, le, un, _, _ = missing_classes("results", "time_millis", whole)
    assert (hi, un) == (2, 0)


# ---------------- mutation tests on real data ----------------
@pytest.fixture(scope="module")
def real():
    return load(INTERIM) if HAVE_DATA else None


def fails(rep):
    return [c["check"] for c in rep.checks if c["status"] == "FAIL"]


@needs_data
def test_real_races_and_results_pass(real):
    rep = Report()
    validate_races(real, rep)
    validate_results(real, rep)
    assert fails(rep) == []


@needs_data
def test_duplicate_race_is_detected(real):
    t = dict(real)
    t["races"] = pd.concat([real["races"], real["races"].iloc[[5]]], ignore_index=True)
    rep = Report()
    validate_races(t, rep)
    assert {"race count", "race_id unique", "(season, round) unique"} <= set(fails(rep))


@needs_data
def test_missing_driver_id_is_detected(real):
    t = dict(real)
    res = real["results"].copy()
    res.loc[10, "driver_id"] = None
    t["results"] = res
    rep = Report()
    validate_results(t, rep)
    assert "missing driver_id" in fails(rep)


@needs_data
def test_tampered_policy_column_is_detected(real):
    t = dict(real)
    rp = real["results_policy"].copy()
    idx = rp.index[rp["grid_status"] == "pit_lane_start"][0]
    rp.loc[idx, "grid_status"] = "given"                  # break P-04 in one row
    t["results_policy"] = rp
    rep = Report()
    validate_results(t, rep)
    assert "policy columns reproduce from raw data (src/policies.py)" in fails(rep)


@needs_data
def test_removed_weather_row_breaks_cross_table(real):
    t = dict(real)
    t["weather_race"] = real["weather_race"].iloc[1:].copy()
    cross = cross_tables(t).set_index("relation")
    assert cross.loc["races <-> weather_race", "status"] == "FAIL"
    assert cross.loc["races <-> weather_race", "left_only"] == 1


# ---------------- full run ----------------
@needs_data
def test_full_run_passes_is_read_only_and_reproducible(tmp_path):
    config = load_config()
    digest = lambda paths: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    inputs = [INTERIM / f"{n}.csv" for n in TABLES]
    before = digest(inputs)
    out1, out2 = tmp_path / "one", tmp_path / "two"
    out1.mkdir(), out2.mkdir()
    first = run(INTERIM, out1, config)
    run(INTERIM, out2, config)
    assert digest(inputs) == before                                       # read-only
    assert digest(sorted(out1.glob("*.csv"))) == digest(sorted(out2.glob("*.csv")))   # identical outputs
    summary = first["raw_validation_summary"]
    assert (summary["status"] == "FAIL").sum() == 0
    assert set(summary["status"]) <= {"PASS", "INFO"}
