"""
Tests for the approved historical policies P-01 ... P-11 (src/policies.py).

Offline tests use small hand-made races in the real interim column layout.
The final test checks the real annotated data if data/interim exists
(skipped otherwise, e.g. on a fresh clone before collection).
"""

import pandas as pd
import pytest

from src.collect import flatten_qualifying, flatten_results
from src.config import PROJECT_ROOT
from src.policies import (POLICY_RESULT_COLUMNS, annotate_races, annotate_results, effective_category,
                          grid_status, is_dnf, result_category_detail)


def race(season, rnd, name="Test GP", circuit="monza", **records):
    return {"season": season, "round": str(rnd), "raceName": name, "date": f"{season}-06-01",
            "Circuit": {"circuitId": circuit, "circuitName": circuit,
                        "Location": {"lat": "0", "long": "0", "locality": "x", "country": "y"}}, **records}


def rec(pos, driver, text=None, status="Finished", laps="50", grid="1", car="1", team="ferrari"):
    return {"number": car, "position": str(pos), "positionText": text or str(pos), "points": "0",
            "grid": grid, "laps": laps, "status": status,
            "Driver": {"driverId": driver, "givenName": driver, "familyName": "X"},
            "Constructor": {"constructorId": team, "name": team.title()}}


RACES = [
    # 1951 French GP style: shared win (fangio + fagioli in car 8), fangio also has a second car (car 2, retired)
    race("1951", 4, Results=[rec(1, "fangio", car="8"), rec(1, "fagioli", car="8"), rec(2, "ascari"),
                             rec(3, "fangio", "R", "Engine", "20", car="2"),
                             rec(4, "moss", "W", "Withdrew", "0", grid="0"),              # did not start
                             rec(5, "gonzalez", "W", "Withdrew", "12", grid="7")]),        # ambiguous: W with laps
    race("1954", 2, name="Indianapolis 500", circuit="indianapolis",
         Results=[rec(1, "vukovich"), rec(2, "hanks", "R", "Engine", "30"), rec(3, "force", "R", "Engine", "35")]),
    race("2005", 9, name="United States Grand Prix", circuit="indianapolis",
         Results=[rec(1, "schumacher"), rec(2, "barrichello", "R", "Tyre", "10")]),
    race("2016", 6, Results=[rec(1, "hamilton"), rec(2, "sainz", grid="0", laps="78"),       # pit-lane start
                             rec(3, "perez", "17", "Engine", "49"),                          # classified, not running
                             rec(4, "russell", "D", "Disqualified", "0", grid="0"),          # started cat., 0 laps
                             rec(5, "frentzen", "5", "Withdrew", "74")]),                    # ambiguous classified
]
QUALI = [race("2016", 6, QualifyingResults=[{"number": "44", "position": "1", "Driver": {"driverId": "hamilton"},
                                             "Constructor": {"constructorId": "mercedes"}, "Q1": "1:30"}])]


@pytest.fixture(scope="module")
def annotated():
    results = flatten_results(RACES)
    original = results.copy()
    out = annotate_results(results, flatten_qualifying(QUALI))
    return original, out.set_index(out["race_id"] + "/" + out["driver_id"] + "/" + out["driver_number"])


def row(out, key):
    return out.loc[key]


def test_original_columns_and_rows_are_unchanged(annotated):
    """P-05 / general: every original column and every row is kept exactly."""
    original, out = annotated
    assert len(out) == len(original)
    pd.testing.assert_frame_equal(out[original.columns].reset_index(drop=True), original)
    assert list(out.columns) == list(original.columns) + POLICY_RESULT_COLUMNS


def test_p01_multiple_entries_primary_is_best_placed_started(annotated):
    _, out = annotated
    assert row(out, "1951_04/fangio/8")["entry_count"] == 2
    assert row(out, "1951_04/fangio/8")["is_primary_entry"]          # P1 in car 8
    assert not row(out, "1951_04/fangio/2")["is_primary_entry"]      # retired in car 2: kept, not primary
    assert not row(out, "1951_04/fangio/2")["in_model_population"]
    assert row(out, "1951_04/ascari/1")["entry_count"] == 1


def test_p02_shared_position_flag_and_winner_metric_exclusion(annotated):
    _, out = annotated
    assert row(out, "1951_04/fangio/8")["shared_position"] and row(out, "1951_04/fagioli/8")["shared_position"]
    assert row(out, "1951_04/fagioli/8")["position"] == "1"           # official position kept
    assert row(out, "1951_04/fagioli/8")["in_model_population"]       # not deleted
    races = annotate_races(pd.DataFrame({"season": ["1951", "2016"], "round": ["4", "6"],
                                         "race_name": ["Test GP", "Test GP"]}), out.reset_index(drop=True))
    assert races.set_index("season")["exclude_from_winner_metrics"].to_dict() == {"1951": True, "2016": False}


def test_p03_missing_qualifying_kept_with_indicators(annotated):
    _, out = annotated
    assert row(out, "2016_06/hamilton/1")["race_has_qualifying"] and row(out, "2016_06/hamilton/1")["has_qualifying_row"]
    assert row(out, "2016_06/sainz/1")["race_has_qualifying"] and not row(out, "2016_06/sainz/1")["has_qualifying_row"]
    assert not row(out, "1951_04/ascari/1")["race_has_qualifying"]
    assert row(out, "1951_04/ascari/1")["in_model_population"]       # race without qualifying is not dropped


def test_p04_grid_zero(annotated):
    _, out = annotated
    assert row(out, "2016_06/sainz/1")["grid_status"] == "pit_lane_start"
    assert row(out, "2016_06/sainz/1")["grid"] == "0"                 # raw value preserved
    assert row(out, "2016_06/russell/1")["grid_status"] == "unknown"  # started category, 0 laps
    assert row(out, "1951_04/moss/1")["grid_status"] == "non_starter"
    assert grid_status("5", True, "50") == "given"
    assert grid_status("7", False, "0") == "non_starter"              # DNS with a grid value


def test_p05_categories_from_text_status_and_laps():
    assert result_category_detail("R", "Engine", "20") == "started_not_classified"
    assert result_category_detail("W", "Engine", "0") == "did_not_start"
    assert result_category_detail("17", "Engine", "49") == "classified_not_running"
    assert result_category_detail("20", "Disqualified", "44") == "disqualified"
    assert result_category_detail("12", "+1 Lap", "56") == "finished"


def test_p06_dnf_is_started_not_classified_only(annotated):
    _, out = annotated
    assert is_dnf("started_not_classified")
    assert not is_dnf("classified_not_running") and not is_dnf("disqualified") and not is_dnf("finished")
    perez = row(out, "2016_06/perez/1")
    assert perez["result_category"] == "classified_not_running" and perez["retired_while_classified"]
    assert not is_dnf(perez["result_category"])
    assert "target_dnf" not in out.columns                           # no target generation in Step 5.6


def test_p07_ambiguous_rows_are_started_and_flagged(annotated):
    _, out = annotated
    gonzalez, frentzen = row(out, "1951_04/gonzalez/1"), row(out, "2016_06/frentzen/1")
    assert gonzalez["ambiguous_status"] and gonzalez["started"]
    assert gonzalez["result_category_detail"] == "ambiguous_withdrawn_after_start"
    assert gonzalez["result_category"] == "started_not_classified"
    assert frentzen["ambiguous_status"] and frentzen["result_category"] == "classified_not_running"
    assert effective_category("finished") == "finished"


def test_p08_model_population(annotated):
    _, out = annotated
    pop = out[out["in_model_population"]]
    assert "1951_04/moss/1" not in pop.index                           # non-starter
    assert not pop["is_indianapolis_500"].any()
    assert not pop.duplicated(["season", "round", "driver_id"]).any()  # one row per driver per race
    assert "1951_04/gonzalez/1" in pop.index                           # ambiguous, treated as started


def test_p09_constructor_id_is_not_merged(annotated):
    original, out = annotated
    assert (out["constructor_id"].values == original["constructor_id"].values).all()
    assert not any("lineage" in c for c in out.columns)


def test_p10_order_exception_flag_without_reordering(annotated):
    original, out = annotated
    # 1954 Indy: hanks (P2, 30 laps) above force (P3, 35 laps) -> inversion, both rows flagged
    assert row(out, "1954_02/hanks/1")["laps_order_exception"] and row(out, "1954_02/force/1")["laps_order_exception"]
    assert not row(out, "2005_09/barrichello/1")["laps_order_exception"]
    assert (out["position"].values == original["position"].values).all()  # order untouched


def test_p11_indianapolis_by_exact_race_name(annotated):
    _, out = annotated
    assert row(out, "1954_02/vukovich/1")["is_indianapolis_500"]
    assert not row(out, "1954_02/vukovich/1")["in_model_population"]
    assert not row(out, "2005_09/schumacher/1")["is_indianapolis_500"]      # same circuit_id, kept
    assert row(out, "2005_09/schumacher/1")["in_model_population"]


# ----------------------------------------------------------------------
# Real data (skipped if the annotated file has not been built)
# ----------------------------------------------------------------------
REAL = PROJECT_ROOT / "data" / "interim" / "results_policy.csv"


@pytest.mark.skipif(not REAL.exists(), reason="run python -m src.policies first")
def test_real_annotated_results_match_step_55_analysis():
    real = pd.read_csv(REAL, dtype=str, keep_default_na=False, na_values=[""])
    raw = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "results.csv", dtype=str,
                      keep_default_na=False, na_values=[""])
    flag = lambda c: real[c] == "True"
    pd.testing.assert_frame_equal(real[raw.columns], raw)                   # original columns identical
    assert len(real) == 25873
    assert flag("multi_entry_driver").sum() == 172                           # P-01
    assert flag("shared_position").sum() == 228                              # P-02
    assert real.loc[flag("race_has_shared_win"), "race_id"].nunique() == 3
    assert flag("pit_lane_start").sum() == 48 and (real["grid_status"] == "unknown").sum() == 17   # P-04
    assert real["status"].nunique() == 136                                   # P-05
    assert flag("ambiguous_status").sum() == 24                              # P-07
    assert flag("laps_order_exception").sum() == 16                          # P-10: 8 pairs
    assert flag("is_indianapolis_500").sum() == 405                          # P-11
    pop = real[flag("in_model_population")]
    assert not pop.duplicated(["season", "round", "driver_id"]).any()        # P-08
    assert not flag("is_indianapolis_500")[pop.index].any()
    assert (pop["result_category"] != "did_not_start").all()
