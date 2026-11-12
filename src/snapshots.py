"""
Driver and constructor performance snapshots for the custom race simulator (D-019).

A snapshot "driver D, season S" is defined as the feature values D would have had for a
hypothetical EXTRA race appended after the final race of season S. They are computed by
running the training feature code (`build_features`) on the real history up to S plus one
pseudo race with no outcome, so a snapshot means exactly what the training features mean:
form over the last races, career rates up to the end of S, season points share at the end of S.

Outputs (data/processed/):
    driver_snapshots.csv       one row per (driver, season) with >= 1 start in that season
    constructor_snapshots.csv  one row per (constructor, season)
    circuits.csv               circuit catalogue with pole-win rate over all history

Run:  python -m src.snapshots   (about 2 minutes)
"""

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.features import build_features, load_features, load_master_base

DRIVER_COLS = ["driver_last5_avg_finish_pct", "driver_last10_avg_finish_pct", "driver_avg_positions_gained_last10",
               "driver_dnf_rate_last10", "driver_win_rate", "driver_podium_rate", "driver_career_starts",
               "driver_age_years", "driver_points_share_before_race", "driver_champ_pos_before_race",
               "driver_wins_season_before_race", "driver_teammate_finish_gap_last10",
               "driver_teammate_grid_gap_last10", "driver_teammate_beat_rate"]
CONSTRUCTOR_COLS = ["constructor_last5_avg_finish_pct", "constructor_win_rate_last20", "constructor_podium_rate_last20",
                    "constructor_dnf_rate_last10", "constructor_points_share_before_race",
                    "constructor_champ_pos_before_race", "constructor_last5_avg_grid_pct",
                    "constructor_last5_best_grid_pct", "constructor_last5_best_finish_pct"]
SNAP_CIRCUIT = "__snapshot__"


def _pseudo_race(mb, season):
    """One outcome-free row per model-population driver of `season`, dated after its last race."""
    pop = mb[mb["in_model_population"] & (mb["season"] == season)].sort_values("race_date")
    last = pop.groupby("driver_id").tail(1).copy()        # each driver's last entry (team at season end)
    last["race_id"] = f"{season}_snap"
    last["round"] = pop["round"].max() + 1
    last["race_date"] = pop["race_date"].max() + pd.Timedelta(days=1)
    last["circuit_id"] = SNAP_CIRCUIT
    last["field_size"] = len(last)
    nan_cols = ["official_position", "target_finish_position", "target_winner", "target_podium", "finish_pct",
                "grid_position", "grid_raw", "qualifying_position", "q1_s", "q2_s", "q3_s", "quali_best_s",
                "sprint_position", "sprint_points", "laps", "time_millis"]
    last[nan_cols] = np.nan
    last[[c for c in last.columns if c.startswith("wx_") and c != "wx_available"]] = np.nan
    last["target_dnf"] = np.nan
    last["points"] = 0.0
    last["grid_status"] = "given"
    last["pit_lane_start"] = False
    return last


def build_snapshots(mb):
    drivers, constructors = [], []
    for season in sorted(mb.loc[mb["in_model_population"], "season"].unique()):
        hist = mb[mb["season"] <= season]
        f = build_features(pd.concat([hist, _pseudo_race(mb, season)], ignore_index=True))
        snap = f[f["race_id"] == f"{season}_snap"].copy()
        n = mb[mb["in_model_population"] & (mb["season"] == season)].groupby("driver_id").size()
        snap["season_starts"] = snap["driver_id"].map(n)
        drivers.append(snap[["driver_id", "season", "driver_code", "driver_first_name", "driver_last_name",
                             "constructor_id", "constructor_name", "season_starts"] + DRIVER_COLS])
        constructors.append(snap.groupby("constructor_id").agg(
            season=("season", "first"), constructor_name=("constructor_name", "first"),
            drivers=("driver_id", lambda s: ",".join(sorted(s))),
            **{c: (c, "first") for c in CONSTRUCTOR_COLS}).reset_index())
    return pd.concat(drivers, ignore_index=True), pd.concat(constructors, ignore_index=True)


def circuit_catalogue(mf):
    r = mf.groupby("race_id").agg(circuit_id=("circuit_id", "first"), circuit_name=("circuit_name", "first"),
                                  country=("country", "first"), race_name=("race_name", "first"),
                                  season=("season", "first"),
                                  pole_won=("target_winner", lambda s: s[mf.loc[s.index, "grid_position"] == 1].max()))
    r = r.sort_values("season")
    return r.groupby("circuit_id").agg(
        circuit_name=("circuit_name", "last"), country=("country", "last"), race_name=("race_name", "last"),
        races=("season", "size"), first_season=("season", "min"), last_season=("season", "max"),
        circuit_pole_win_rate=("pole_won", "mean")).reset_index()


def main():
    processed = PROJECT_ROOT / load_config()["paths"]["data_processed"]
    d, c = build_snapshots(load_master_base())
    d.to_csv(processed / "driver_snapshots.csv", index=False)
    c.to_csv(processed / "constructor_snapshots.csv", index=False)
    circuit_catalogue(load_features()).to_csv(processed / "circuits.csv", index=False)
    print(f"driver snapshots {len(d)}, constructor snapshots {len(c)}")


if __name__ == "__main__":
    main()
