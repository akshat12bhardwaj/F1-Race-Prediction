"""Leakage tests T1-T6 (blueprint §19.5)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import PROJECT_ROOT, load_config
from src.features import DRIVER_HISTORY, FEATURES, SCENARIO_EXTRA, build_features, load_features, load_master_base
from src.splits import season_split

BLANK = ["official_position", "points", "target_finish_position", "target_winner", "target_podium",
         "target_dnf", "finish_pct", "laps", "status", "position_text", "time_millis"]


@pytest.fixture(scope="module")
def mb():
    return load_master_base()


@pytest.fixture(scope="module")
def mf():
    return load_features()


def _same(a, b):
    cols = FEATURES + [c for c in SCENARIO_EXTRA if c not in FEATURES]
    a, b = a[cols].to_numpy(float), b[cols].to_numpy(float)
    return np.allclose(a, b, equal_nan=True)


def test_t1_recompute_from_past_only(mb, mf):
    """Rebuild features using only races before the target race (its outcomes blanked)."""
    rng = np.random.default_rng(42)
    races = rng.choice(mf.loc[mf["season"] >= 1960, "race_id"].unique(), 8, replace=False)
    for rid in races:
        date = mb.loc[mb["race_id"] == rid, "race_date"].iloc[0]
        past = mb[(mb["race_date"] < date) | (mb["race_id"] == rid)].copy()
        cur = past["race_id"] == rid
        past.loc[cur, BLANK] = np.nan
        rebuilt = build_features(past)
        a = rebuilt[rebuilt["race_id"] == rid].sort_values("driver_id")
        b = mf[mf["race_id"] == rid].sort_values("driver_id")
        assert len(a) == len(b) and _same(a, b), rid


def test_t2_perturbation(mb, mf):
    """Changing one race's outcomes leaves its own features unchanged and changes later races."""
    rid = "2010_05"
    m = mb.copy()
    cur = m["race_id"] == rid
    m.loc[cur & m["in_model_population"], "target_finish_position"] = (
        m.loc[cur & m["in_model_population"], "target_finish_position"].max() + 1
        - m.loc[cur & m["in_model_population"], "target_finish_position"])
    m.loc[cur, "points"] = m.loc[cur, "points"][::-1].to_numpy()
    m.loc[cur, "target_dnf"] = 1 - m.loc[cur, "target_dnf"]
    rebuilt = build_features(m)
    a = rebuilt[rebuilt["race_id"] == rid].sort_values("driver_id")
    b = mf[mf["race_id"] == rid].sort_values("driver_id")
    assert _same(a, b)
    nxt = rebuilt[rebuilt["race_id"] == "2010_06"].sort_values("driver_id")
    assert not _same(nxt, mf[mf["race_id"] == "2010_06"].sort_values("driver_id"))


def test_t3_cold_start(mf):
    first = mf[mf.groupby("driver_id").cumcount() == 0]
    assert first[DRIVER_HISTORY].isna().all().all()
    assert (first["driver_career_starts"] == 0).all()


def test_t4_suspicion_single_feature_auc(mf):
    tr = mf[mf["season"] <= load_config()["split"]["train_end"]]
    y = tr["target_winner"]
    for f in FEATURES:
        if f.startswith(("grid", "qualifying", "quali", "front_row", "sprint")):
            continue
        x = tr[f].fillna(tr[f].median())
        if x.nunique() < 2:
            continue
        auc = roc_auc_score(y, x)
        assert max(auc, 1 - auc) < 0.95, (f, auc)


def test_t5_split_disjoint_and_ordered(mf):
    tr, va, te = season_split(mf)
    assert not set(tr["race_id"]) & set(va["race_id"]) and not set(va["race_id"]) & set(te["race_id"])
    assert tr["race_date"].max() < va["race_date"].min() < te["race_date"].min()


def test_t6_column_denylist():
    deny = ("position_text", "points", "laps", "status", "time", "fastest", "pit_stop", "target_", "official")
    feats = json.loads((PROJECT_ROOT / "data/processed/feature_list_v1.json").read_text())["features"]
    assert feats == FEATURES
    bad = [f for f in feats if any(d in f for d in deny) and f not in
           ("driver_points_share_before_race", "constructor_points_share_before_race", "pit_lane_start",
            "wx_time_imputed")]
    assert not bad, bad
    assert not any(f.startswith("wx_race_") for f in feats), "race-window weather is oracle-only"
