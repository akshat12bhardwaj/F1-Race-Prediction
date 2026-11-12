"""Phase 1: master_base integrity (blueprint §18) on the real processed file."""

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.dataset import build_master_base, check_master_base, parse_lap_time


@pytest.fixture(scope="module")
def mb():
    return build_master_base(PROJECT_ROOT / "data" / "interim")


def test_parse_lap_time():
    assert parse_lap_time("1:29.179") == pytest.approx(89.179)
    assert parse_lap_time("53.904") == pytest.approx(53.904)
    assert np.isnan(parse_lap_time("")) and np.isnan(parse_lap_time(None)) and np.isnan(parse_lap_time("DNF"))


def test_integrity_assertions_pass(mb):
    stats = check_master_base(mb)
    assert stats["model_population_rows"] == 25066 and stats["pit_lane_starts"] == 48


def test_saved_file_matches_build(mb):
    saved = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "master_base.csv", low_memory=False)
    assert len(saved) == len(mb)
    assert (saved["row_id"].values == mb["row_id"].values).all()
    np.testing.assert_allclose(saved["target_finish_position"], mb["target_finish_position"])


def test_starters_vs_non_starters(mb):
    ns = mb[~mb["started"]]
    assert ns["target_finish_position"].isna().all() and ns["grid_position"].isna().all()
    pop = mb[mb["in_model_population"]]
    assert pop["started"].all() and pop["is_primary_entry"].all() and not pop["is_indianapolis_500"].any()
    # DNF (P-06) and finisher flags are mutually exclusive
    assert not (pop["is_finisher"] & (pop["target_dnf"] == 1)).any()


def test_pit_lane_grid_is_field_size_plus_one(mb):
    pit = mb[mb["grid_status"] == "pit_lane_start"]
    assert (pit["grid_position"] == pit["field_size"] + 1).all()


def test_known_race_2021_abu_dhabi(mb):
    r = mb[(mb["race_id"] == "2021_22") & mb["in_model_population"]].set_index("driver_id")
    assert r.loc["max_verstappen", "target_winner"] == 1 and r.loc["hamilton", "target_finish_position"] == 2
    assert r.loc["max_verstappen", "grid_position"] == 1
    assert r.loc["max_verstappen", "qualifying_position"] == 1
