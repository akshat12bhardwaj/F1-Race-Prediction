"""Teammate-network ratings: leakage safety and basic behaviour."""

import numpy as np
import pandas as pd

from src.driver_ratings import asof_table, attach, fit_ratings, load_ratings
from src.features import load_features


def test_asof_ratings_use_only_past_seasons():
    """Ratings 'as of 1965' must be identical whether or not later seasons exist in the data."""
    mf = load_features()
    full = load_ratings()
    part = asof_table(mf[mf["season"] <= 1965])
    a = full[full["asof_season"] == 1965].set_index("driver_id").sort_index()
    b = part[part["asof_season"] == 1965].set_index("driver_id").sort_index()
    assert list(a.index) == list(b.index)
    assert np.allclose(a[["net_quali_skill", "net_race_skill"]], b[["net_quali_skill", "net_race_skill"]], equal_nan=True)


def test_rating_recovers_consistent_teammate_ordering():
    # A beats B by 1, B beats C by 1 (never teammates A-C): network must order A > B > C
    pairs = pd.DataFrame({"season": [2000] * 6, "d_i": ["a", "a", "a", "b", "b", "b"],
                          "d_j": ["b", "b", "b", "c", "c", "c"], "gap_quali": [-1.0] * 6, "gap_race": [-1.0] * 6})
    r = fit_ratings(pairs, "gap_quali").droplevel("season")
    assert r["a"] > r["b"] > r["c"]


def test_training_rows_use_previous_season_ratings():
    mf = load_features()
    x = attach(mf[mf["race_id"] == "2021_01"], load_ratings(), -1).set_index("driver_id")
    r = load_ratings()
    expected = r[(r["driver_id"] == "hamilton") & (r["asof_season"] == 2020)]["net_quali_skill"].iloc[0]
    assert x.loc["hamilton", "net_quali_skill"] == expected
    assert np.isnan(x.loc["tsunoda", "net_quali_skill"])            # 2021 rookie: no rating before his debut
