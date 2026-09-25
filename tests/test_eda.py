"""
Tests for src/eda.py (Step 8).

- Unit tests of small helpers and of the variable-role (leakage) labels.
- Real-data tests (skipped on a fresh clone): documented counts, no targets,
  no imputation, deterministic tables, read-only inputs.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.eda import build_tables, era_of, grid_finish_matrix, load, run, variable_roles

INTERIM = PROJECT_ROOT / "data" / "interim"
TABLES = PROJECT_ROOT / "outputs" / "tables"
HAVE_DATA = (INTERIM / "results_policy.csv").exists() and (TABLES / "raw_anomalies.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="interim data / Step 7 outputs not built")


# ---------------- unit tests ----------------
def test_era_boundaries():
    assert [era_of(s) for s in ["1950", "1969", "1970", "1989", "1990", "2009", "2010", "2025"]] == \
        ["1950-1969", "1950-1969", "1970-1989", "1970-1989", "1990-2009", "1990-2009", "2010-2025", "2010-2025"]


def test_variable_roles_label_outcomes_as_post_race():
    roles = variable_roles().set_index(["table", "column"])["timing"]
    for col in ["position", "position_text", "points", "laps", "status", "time_millis", "fastest_lap_rank"]:
        assert roles[("results", col)] == "POST-RACE OUTCOME"
    assert roles[("results_policy", "result_category")] == "POST-RACE OUTCOME"
    assert roles[("results_policy", "is_primary_entry")].startswith("POST-RACE")     # uses position
    assert roles[("results", "grid")] == "pre-race at T0"
    assert roles[("qualifying", "qualifying_position")] == "pre-race at T0"
    assert roles[("weather_race", "wx_race_temp_mean_c")].startswith("DURING RACE")
    assert roles[("weather_race", "wx_prestart_temp_mean_c")].startswith("observed before T0")
    assert roles[("results", "permanent_number")] == "not historical"


def test_grid_finish_matrix_rows_are_shares():
    P = pd.DataFrame({"era": ["2010-2025"] * 4, "grid_n": [1, 1, 2, 2], "position_n": [1, 2, 1, 25]})
    m = grid_finish_matrix(P, "2010-2025", n=3)
    assert list(m.columns) == [1, 2, 3, 4]                         # 4 = "n+1 or worse"
    assert m.loc[1, 1] == 0.5 and m.loc[1, 2] == 0.5
    assert m.loc[2, 4] == 0.5                                       # position 25 folded into 4+
    assert np.isnan(m.loc[3]).all()                                 # no starters from grid 3


# ---------------- real data ----------------
@pytest.fixture(scope="module")
def tables():
    return build_tables(load(INTERIM, TABLES))[1] if HAVE_DATA else None


@needs_data
def test_documented_counts(tables):
    ov = tables["eda_dataset_overview"].set_index("metric")["value"]
    assert ov["races"] == 1149 and ov["eligible races"] == 1138
    assert ov["Indianapolis 500 races (excluded, P-11)"] == 11
    assert ov["result rows (all entries)"] == 25873 and ov["model-population rows (P-08)"] == 25066
    assert ov["qualifying rows"] == 10973 and ov["sprint rows"] == 480
    assert ov["weather coverage of eligible races (%)"] == 100.0
    gz = tables["eda_grid_zero"]
    assert gz["total_grid_zero"].sum() == 267 and gz["pit_lane_start"].sum() == 48
    um = tables["eda_unexpected_missing"]
    assert um.loc[um["field"] == "TOTAL", "values"].iloc[0] == 286


@needs_data
def test_no_targets_and_no_imputation(tables):
    for name, df in tables.items():
        assert not any(str(c).lower().startswith("target") for c in df.columns), name
    miss = tables["eda_missingness"].set_index(["table", "field"])
    assert miss.loc[("qualifying", "q3"), "missing"] == 7117          # still missing -> nothing filled
    assert miss.loc[("results", "grid"), "missing"] == 0


@needs_data
def test_tables_are_deterministic_and_inputs_unchanged(tmp_path):
    inputs = sorted(INTERIM.glob("*.csv")) + [TABLES / "raw_anomalies.csv"]
    digest = lambda paths: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = digest(inputs)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    run(INTERIM, a, fig_dir=None, input_tables_dir=TABLES)
    run(INTERIM, b, fig_dir=None, input_tables_dir=TABLES)
    assert digest(inputs) == before
    assert digest(sorted(a.glob("*.csv"))) == digest(sorted(b.glob("*.csv")))
    assert len(list(a.glob("eda_*.csv"))) == 31
