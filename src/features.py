"""
Phase 2 (blueprint Step 10-11, §19-20): leakage-safe pre-race features.

    data/processed/master_features.csv   model population rows, features + targets
    data/processed/feature_list_v1.json  the model feature list (T0-safe)

Rules (§19.4):
- History is built from model-population rows (started, primary entry, not Indy 500),
  sorted by (race_date, race_id).
- Every history feature is aggregate -> shift(1) -> roll, so a race never sees its own
  outcome. Team and circuit features are aggregated per (group, race) FIRST, then shifted,
  so a teammate's current-race result cannot leak.
- Season standings use points from rounds strictly before the current round.
- Current-race columns are only pre-race facts: grid, qualifying, same-weekend sprint,
  field size (number of starters), pre-start weather.
- Race-window weather (wx_race_*) is an oracle proxy: kept for the E6 ablation only and
  NOT in feature_list_v1.

Run:  python -m src.features
"""

import json

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

OUTCOME_COLS = ["official_position", "position_text", "points", "laps", "status", "time_millis",
                "time_text", "target_finish_position", "target_winner", "target_podium", "target_dnf",
                "finish_pct", "result_category", "result_category_detail", "is_finisher", "is_dsq",
                "fastest_lap_rank", "fastest_lap_lap", "fastest_lap_time", "fastest_lap_avg_speed",
                "retired_while_classified", "shared_position", "shared_position_count"]

FEATURE_GROUPS = {
    "grid_quali": ["grid_position", "grid_pct", "front_row", "pit_lane_start", "field_size",
                   "qualifying_position", "quali_gap_to_pole_pct", "quali_missing", "teammate_grid_diff"],
    "sprint": ["sprint_position", "has_sprint"],
    "driver_form": ["driver_prev_finish", "driver_last3_avg_finish", "driver_last5_avg_finish",
                    "driver_last10_avg_finish", "driver_last10_avg_finish_pct",
                    "driver_avg_positions_gained_last10"],
    "driver_career": ["driver_career_starts", "driver_career_avg_finish", "driver_win_rate",
                      "driver_podium_rate", "driver_dnf_rate_last10", "driver_age_years", "is_rookie"],
    "championship": ["driver_points_share_before_race", "driver_champ_pos_before_race",
                     "driver_wins_season_before_race", "constructor_points_share_before_race",
                     "constructor_champ_pos_before_race"],
    "constructor": ["constructor_last5_avg_finish", "constructor_win_rate_last20",
                    "constructor_podium_rate_last20", "constructor_dnf_rate_last10",
                    "constructor_prev_season_rank"],
    "circuit": ["driver_circuit_starts", "driver_circuit_avg_finish", "constructor_circuit_avg_finish",
                "circuit_pole_win_rate", "circuit_dnf_rate"],
    "weather_prestart": ["wx_prestart_temp_mean_c", "wx_prestart_humidity_mean_pct",
                         "wx_prestart_precip_sum_mm", "wx_prestart_cloud_mean_pct",
                         "wx_prestart_wind_speed_mean_kmh", "wx_prestart_weather_code_max",
                         "wx_time_imputed"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
ORACLE_WEATHER = ["wx_race_temp_mean_c", "wx_race_humidity_mean_pct", "wx_race_precip_sum_mm",
                  "wx_race_cloud_mean_pct", "wx_race_wind_speed_mean_kmh", "wx_race_weather_code_max"]
# History features that must be NaN in a driver's first career race (leakage test T3)
DRIVER_HISTORY = ["driver_prev_finish", "driver_last3_avg_finish", "driver_last5_avg_finish",
                  "driver_last10_avg_finish", "driver_last10_avg_finish_pct",
                  "driver_avg_positions_gained_last10", "driver_career_avg_finish", "driver_win_rate",
                  "driver_podium_rate", "driver_dnf_rate_last10", "driver_circuit_avg_finish",
                  "driver_last5_avg_finish_pct", "driver_circuit_avg_finish_pct",
                  "driver_teammate_finish_gap_last10", "driver_teammate_grid_gap_last10", "driver_teammate_beat_rate"]
# Extra era-comparable columns used by the custom-race (scenario) model; same leakage rules.
SCENARIO_EXTRA = ["driver_last5_avg_finish_pct", "driver_circuit_avg_finish_pct",
                  "constructor_last5_avg_finish_pct", "constructor_circuit_avg_finish_pct",
                  "wx_prestart_wet", "circuit_pole_win_rate",
                  # car-neutral driver measures (vs. teammates in the same car) and car qualifying pace (D-021)
                  "driver_teammate_finish_gap_last10", "driver_teammate_grid_gap_last10", "driver_teammate_beat_rate",
                  "constructor_last5_avg_grid_pct", "constructor_circuit_avg_grid_pct",
                  "constructor_last5_best_grid_pct", "constructor_last5_best_finish_pct"]
ID_COLS = ["race_driver_id", "race_id", "season", "round", "race_date", "race_name", "circuit_id",
           "circuit_name", "country", "driver_id", "driver_code", "driver_first_name", "driver_last_name",
           "driver_number", "constructor_id", "constructor_name", "exclude_from_winner_metrics"]
TARGETS = ["target_finish_position", "target_winner", "target_podium", "target_dnf", "finish_pct",
           "official_position", "position_text", "status", "result_category", "points"]


def _past(s, window=None, how="mean"):
    """shift(1) then rolling/expanding: value uses strictly earlier rows of the group."""
    p = s.shift(1)
    r = p.rolling(window, min_periods=1) if window else p.expanding(min_periods=1)
    return getattr(r, how)()


def _hist(df, keys, col, window=None, how="mean"):
    return df.groupby(keys, sort=False)[col].transform(lambda s: _past(s, window, how))


def _standings(points_rows, entity):
    """Points/wins before each (season, round) for one entity column; rounds < current only."""
    per = points_rows.groupby(["season", "round", entity], as_index=False)[["pts", "win"]].sum()
    per = per.sort_values(["season", "round"])
    g = per.groupby(["season", entity])
    per["points_before"] = g["pts"].cumsum() - per["pts"]
    per["wins_before"] = g["win"].cumsum() - per["win"]
    return per[["season", "round", entity, "points_before", "wins_before"]]


def build_features(mb):
    """mb = master_base (all result rows). Returns model-population rows with features."""
    mb = mb.sort_values(["race_date", "race_id", "official_position"], kind="mergesort")
    df = mb[mb["in_model_population"]].copy().reset_index(drop=True)
    fs = df["field_size"].astype(float)
    df["race_order"] = df["race_date"].rank(method="dense").astype(int)

    # ---------- current race, pre-race facts ----------
    df["grid_pct"] = (df["grid_position"] - 1) / (fs - 1)
    df["front_row"] = (df["grid_position"] <= 2).astype(float).where(df["grid_position"].notna())
    df["pit_lane_start"] = df["pit_lane_start"].astype(int)
    pole = df.groupby("race_id")["quali_best_s"].transform("min")
    df["quali_gap_to_pole_pct"] = (df["quali_best_s"] / pole - 1) * 100
    df["quali_missing"] = df["qualifying_position"].isna().astype(int)
    team_grid = df.groupby(["race_id", "constructor_id"])["grid_position"]
    n_team = team_grid.transform("count")
    df["teammate_grid_diff"] = np.where(
        n_team >= 2, df["grid_position"] - (team_grid.transform("sum") - df["grid_position"]) / (n_team - 1),
        np.nan)
    df["has_sprint"] = df["sprint_position"].notna().astype(int)

    # ---------- driver history (one row per driver per race) ----------
    pos = df["target_finish_position"]
    df["_win"] = (pos == 1).astype(float)
    df["_podium"] = (pos <= 3).astype(float)
    df["_gained"] = df["grid_position"] - pos
    d = ["driver_id"]
    df["driver_career_starts"] = df.groupby(d).cumcount()
    df["driver_prev_finish"] = df.groupby(d)["target_finish_position"].shift(1)
    for n in (3, 5, 10):
        df[f"driver_last{n}_avg_finish"] = _hist(df, d, "target_finish_position", n)
    df["driver_last10_avg_finish_pct"] = _hist(df, d, "finish_pct", 10)
    df["driver_last5_avg_finish_pct"] = _hist(df, d, "finish_pct", 5)
    df["driver_avg_positions_gained_last10"] = _hist(df, d, "_gained", 10)
    df["driver_career_avg_finish"] = _hist(df, d, "target_finish_position")
    df["driver_win_rate"] = _hist(df, d, "_win")
    df["driver_podium_rate"] = _hist(df, d, "_podium")
    df["driver_dnf_rate_last10"] = _hist(df, d, "target_dnf", 10)
    df["driver_age_years"] = (df["race_date"] - df["driver_dob"]).dt.days / 365.25
    df["is_rookie"] = (df["driver_career_starts"] < 10).astype(int)

    # ---------- teammate-relative (car-neutral) measures: same race, same car; used only after shift ----------
    team = df.groupby(["race_id", "constructor_id"])
    n_team_cars = team["driver_id"].transform("size")
    for col, tmp in (("finish_pct", "_tm_fin"), ("grid_pct", "_tm_grid")):
        tot, cnt = team[col].transform("sum"), team[col].transform("count")
        mates = (tot - df[col].fillna(0)) / (cnt - df[col].notna())        # mean of the teammates, excluding self
        df[tmp] = (df[col] - mates).where(n_team_cars >= 2)               # negative = ahead of teammates
    beaten = n_team_cars - team["target_finish_position"].rank(method="average")
    df["_tm_beat"] = (beaten / (n_team_cars - 1)).where(n_team_cars >= 2)  # share of teammates finished ahead of
    df["driver_teammate_finish_gap_last10"] = _hist(df, d, "_tm_fin", 10)
    df["driver_teammate_grid_gap_last10"] = _hist(df, d, "_tm_grid", 10)
    df["driver_teammate_beat_rate"] = _hist(df, d, "_tm_beat")

    # ---------- circuit history ----------
    df["driver_circuit_starts"] = df.groupby(["driver_id", "circuit_id"]).cumcount()
    df["driver_circuit_avg_finish"] = _hist(df, ["driver_id", "circuit_id"], "target_finish_position")
    df["driver_circuit_avg_finish_pct"] = _hist(df, ["driver_id", "circuit_id"], "finish_pct")

    # ---------- constructor history: aggregate per (team, race) THEN shift ----------
    tr = df.groupby(["constructor_id", "race_id"], sort=False).agg(
        race_order=("race_order", "first"), season=("season", "first"), circuit_id=("circuit_id", "first"),
        t_finish=("target_finish_position", "mean"), t_finish_pct=("finish_pct", "mean"),
        t_grid_pct=("grid_pct", "mean"), t_best_grid_pct=("grid_pct", "min"), t_best_finish_pct=("finish_pct", "min"),
        t_win=("_win", "max"), t_podium=("_podium", "max"),
        t_dnf=("target_dnf", "mean")).reset_index().sort_values("race_order", kind="mergesort")
    c = ["constructor_id"]
    tr["constructor_last5_avg_finish"] = _hist(tr, c, "t_finish", 5)
    tr["constructor_win_rate_last20"] = _hist(tr, c, "t_win", 20)
    tr["constructor_podium_rate_last20"] = _hist(tr, c, "t_podium", 20)
    tr["constructor_dnf_rate_last10"] = _hist(tr, c, "t_dnf", 10)
    tr["constructor_circuit_avg_finish"] = _hist(tr, ["constructor_id", "circuit_id"], "t_finish")
    tr["constructor_last5_avg_finish_pct"] = _hist(tr, c, "t_finish_pct", 5)
    tr["constructor_circuit_avg_finish_pct"] = _hist(tr, ["constructor_id", "circuit_id"], "t_finish_pct")
    tr["constructor_last5_avg_grid_pct"] = _hist(tr, c, "t_grid_pct", 5)
    # best-placed car per race = what the car could do with its faster driver (car potential, D-021)
    tr["constructor_last5_best_grid_pct"] = _hist(tr, c, "t_best_grid_pct", 5)
    tr["constructor_last5_best_finish_pct"] = _hist(tr, c, "t_best_finish_pct", 5)
    tr["constructor_circuit_avg_grid_pct"] = _hist(tr, ["constructor_id", "circuit_id"], "t_grid_pct")
    keep = ["constructor_id", "race_id", "constructor_last5_avg_finish", "constructor_win_rate_last20",
            "constructor_podium_rate_last20", "constructor_dnf_rate_last10", "constructor_circuit_avg_finish",
            "constructor_last5_avg_finish_pct", "constructor_circuit_avg_finish_pct",
            "constructor_last5_avg_grid_pct", "constructor_circuit_avg_grid_pct",
            "constructor_last5_best_grid_pct", "constructor_last5_best_finish_pct"]
    df = df.merge(tr[keep], on=["constructor_id", "race_id"], how="left", validate="many_to_one")

    # ---------- circuit race-level history ----------
    df["_pole_win"] = df["_win"].where(df["grid_position"] == 1)   # NaN for races without a pole row
    rc = df.groupby("race_id", sort=False).agg(
        race_order=("race_order", "first"), circuit_id=("circuit_id", "first"),
        pole_won=("_pole_win", "max"),
        dnf_share=("target_dnf", "mean")).reset_index().sort_values("race_order", kind="mergesort")
    rc["circuit_pole_win_rate"] = _hist(rc, ["circuit_id"], "pole_won")
    rc["circuit_dnf_rate"] = _hist(rc, ["circuit_id"], "dnf_share")
    df = df.merge(rc[["race_id", "circuit_pole_win_rate", "circuit_dnf_rate"]], on="race_id", how="left")

    # ---------- championship before the race (all entries + sprints, P-01; rounds < current) ----------
    hist_rows = mb[mb["started"] & ~mb["is_indianapolis_500"]]
    pts = hist_rows[["season", "round", "driver_id", "constructor_id", "points", "sprint_points"]].copy()
    pts["pts"] = pts["points"].fillna(0) + pts["sprint_points"].fillna(0)
    pts["win"] = (hist_rows["official_position"] == 1).astype(int)
    for ent, pre in (("driver_id", "driver"), ("constructor_id", "constructor")):
        st = _standings(pts, ent)
        df = df.merge(st.rename(columns={"points_before": f"_{pre}_pb", "wins_before": f"_{pre}_wb"}),
                      on=["season", "round", ent], how="left", validate="many_to_one")
        leader = df.groupby("race_id")[f"_{pre}_pb"].transform("max")
        df[f"{pre}_points_share_before_race"] = (df[f"_{pre}_pb"] / leader).where(leader > 0)
        rank = df.groupby("race_id")[f"_{pre}_pb"].rank(method="min", ascending=False)
        df[f"{pre}_champ_pos_before_race"] = rank.where(leader > 0)
    df["driver_wins_season_before_race"] = df["_driver_wb"]

    # constructor rank in the previous season (derived points; P-09 IDs as-is)
    season_tot = pts.groupby(["season", "constructor_id"])["pts"].sum().reset_index()
    season_tot["constructor_prev_season_rank"] = season_tot.groupby("season")["pts"].rank(
        method="min", ascending=False)
    season_tot["season"] += 1
    df = df.merge(season_tot[["season", "constructor_id", "constructor_prev_season_rank"]],
                  on=["season", "constructor_id"], how="left")

    wcfg = load_config()["weather"]
    # wet = measured pre-start precipitation >= threshold (D-019; reanalysis weather codes >= 51 flag
    # light drizzle in ~31% of starts, far more than genuinely wet races, so codes are not used)
    df["wx_prestart_wet"] = (df["wx_prestart_precip_sum_mm"] >= wcfg["wet_precip_threshold_mm"]).astype(float)         .where(df["wx_prestart_precip_sum_mm"].notna())
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    cols = ID_COLS + FEATURES + [c for c in SCENARIO_EXTRA if c not in FEATURES] + ORACLE_WEATHER + TARGETS
    return df[cols].sort_values(["race_date", "race_id", "target_finish_position"], kind="mergesort") \
        .reset_index(drop=True)


def load_master_base(path=None):
    path = path or PROJECT_ROOT / load_config()["paths"]["data_processed"] / "master_base.csv"
    mb = pd.read_csv(path, low_memory=False, parse_dates=["race_date", "driver_dob"])
    return mb


def load_features(path=None):
    path = path or PROJECT_ROOT / load_config()["paths"]["data_processed"] / "master_features.csv"
    return pd.read_csv(path, parse_dates=["race_date"], dtype={"driver_code": str})


def main():
    processed = PROJECT_ROOT / load_config()["paths"]["data_processed"]
    mf = build_features(load_master_base())
    assert mf["race_driver_id"].is_unique and len(mf) == 25066
    mf.to_csv(processed / "master_features.csv", index=False)
    (processed / "feature_list_v1.json").write_text(json.dumps(
        {"features": FEATURES, "groups": FEATURE_GROUPS, "oracle_weather_ablation_only": ORACLE_WEATHER},
        indent=2))
    print(f"rows {len(mf)}, features {len(FEATURES)}")
    print(mf[FEATURES].isna().mean().round(3).sort_values(ascending=False).head(12).to_string())


if __name__ == "__main__":
    main()
