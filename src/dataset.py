"""
Phase 1 (blueprint Step 9): typed, integrated model base table.

Reads the validated interim tables (never modified) and writes
    data/processed/master_base.csv   one row per result row (all 25,873), typed

Contents per row: race metadata, driver/constructor identity, numeric grid/position/
points/laps, parsed qualifying times, same-weekend sprint result, race weather
(pre-start window = T0-safe; race window = oracle proxy, ablation only), the approved
policy columns, and the targets.

Target (D-016): target_finish_position = rank of the official position among the
model-population starters of the race (1..k; co-driver shared positions keep a tie,
P-02). The official position is kept as `official_position`.

Run:  python -m src.dataset
"""

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.policies import is_dnf

WX_COLS = ["temp_mean_c", "humidity_mean_pct", "precip_sum_mm", "rain_sum_mm", "cloud_mean_pct",
           "wind_speed_mean_kmh", "weather_code_max"]


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def _bool(s):
    return s.map({"True": True, "False": False}).astype(bool)


def parse_lap_time(value):
    """'1:29.179' -> 89.179 s; '53.904' -> 53.904 s; empty/invalid -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    try:
        parts = value.strip().split(":")
        secs = float(parts[-1])
        return secs + (60 * int(parts[0]) if len(parts) == 2 else 0)
    except ValueError:
        return np.nan


def build_master_base(interim_dir):
    r = _read(interim_dir / "results_policy.csv")
    races = _read(interim_dir / "races_policy.csv")
    q = _read(interim_dir / "qualifying.csv")
    s = _read(interim_dir / "sprints.csv")
    w = _read(interim_dir / "weather_race.csv")
    n_in = len(r)

    out = r.copy()
    for c in ["season", "round", "grid", "position", "points", "laps", "driver_number"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.rename(columns={"position": "official_position", "grid": "grid_raw"})
    out["race_date"] = pd.to_datetime(out["race_date"])
    out["driver_dob"] = pd.to_datetime(out["driver_dob"], errors="coerce")
    for c in ["is_primary_entry", "multi_entry_driver", "shared_position", "race_has_shared_win",
              "race_has_qualifying", "has_qualifying_row", "pit_lane_start", "ambiguous_status",
              "started", "retired_while_classified", "laps_order_exception", "is_indianapolis_500",
              "in_model_population"]:
        out[c] = _bool(out[c])

    # ---- race-level metadata (circuit coordinates, winner-metric exclusion) ----
    rmeta = races[["race_id", "circuit_lat", "circuit_lon", "exclude_from_winner_metrics"]].copy()
    rmeta["exclude_from_winner_metrics"] = _bool(rmeta["exclude_from_winner_metrics"])
    out = out.merge(rmeta, on="race_id", how="left", validate="many_to_one")

    # ---- qualifying: unique per (race, driver) [F-003] ----
    qq = q[["race_id", "driver_id", "qualifying_position", "q1", "q2", "q3"]].copy()
    assert not qq.duplicated(["race_id", "driver_id"]).any()
    qq["qualifying_position"] = pd.to_numeric(qq["qualifying_position"])
    for c in ["q1", "q2", "q3"]:
        qq[f"{c}_s"] = qq[c].map(parse_lap_time)
    qq["quali_best_s"] = qq[["q1_s", "q2_s", "q3_s"]].min(axis=1)
    out = out.merge(qq.drop(columns=["q1", "q2", "q3"]), on=["race_id", "driver_id"], how="left",
                    validate="many_to_one")

    # ---- sprint (Saturday, before the GP): pre-race for the GP; points for standings ----
    ss = s[["race_id", "driver_id", "position", "points"]].rename(
        columns={"position": "sprint_position", "points": "sprint_points"})
    assert not ss.duplicated(["race_id", "driver_id"]).any()
    ss[["sprint_position", "sprint_points"]] = ss[["sprint_position", "sprint_points"]].apply(pd.to_numeric)
    out = out.merge(ss, on=["race_id", "driver_id"], how="left", validate="many_to_one")
    # a multi-entry driver must not receive sprint points twice (sprints are 2021+, no multi-entries)
    assert not (out["sprint_points"].notna() & out["multi_entry_driver"]).any()

    # ---- weather: pre-start (T0-safe) and race window (oracle proxy) ----
    wcols = ["race_id", "start_utc", "start_time_source", "model_use_allowed"]
    wcols += [f"wx_prestart_{c}" for c in WX_COLS] + [f"wx_race_{c}" for c in WX_COLS]
    ww = w[wcols].copy()
    for c in wcols[4:]:
        ww[c] = pd.to_numeric(ww[c])
    ww["model_use_allowed"] = _bool(ww["model_use_allowed"])
    ww = ww.rename(columns={"model_use_allowed": "wx_available"})
    ww["wx_time_imputed"] = (ww["start_time_source"] == "imputed_local_14h").astype(int)
    out = out.merge(ww, on="race_id", how="left", validate="many_to_one")
    out["wx_available"] = out["wx_available"].fillna(False).astype(bool)

    # ---- identifiers ----
    out["row_id"] = (out["race_id"] + "_" + out["driver_id"] + "_"
                     + out["driver_number"].astype("Int64").astype(str))
    out["race_driver_id"] = out["race_id"] + "_" + out["driver_id"]

    # ---- population-level derived columns (model population = P-08) ----
    pop = out["in_model_population"]
    out["field_size"] = out["race_id"].map(out[pop].groupby("race_id").size()).astype("Int64")
    out["target_finish_position"] = np.nan
    out.loc[pop, "target_finish_position"] = (
        out[pop].groupby("race_id")["official_position"].rank(method="min"))
    # Grid (P-04): given -> grid; pit-lane start -> field_size + 1; unknown / non-starter -> NaN
    out["grid_position"] = np.where(out["grid_status"] == "given", out["grid_raw"], np.nan)
    pit = out["grid_status"] == "pit_lane_start"
    out.loc[pit, "grid_position"] = out.loc[pit, "field_size"].astype(float) + 1
    out["is_finisher"] = out["result_category"].isin(["finished", "classified_not_running"])
    out["is_dsq"] = out["result_category"] == "disqualified"
    out["target_dnf"] = out["result_category"].map(is_dnf).astype(int)
    tp = out["target_finish_position"]
    out["target_winner"] = np.where(pop, (tp == 1).astype(float), np.nan)
    out["target_podium"] = np.where(pop, (tp <= 3).astype(float), np.nan)
    out["finish_pct"] = (tp - 1) / (out["field_size"].astype(float) - 1)

    out = out.sort_values(["race_date", "race_id", "official_position", "driver_number"], kind="mergesort")
    assert len(out) == n_in, "row count changed during integration"
    return out.reset_index(drop=True)


def check_master_base(mb):
    """Integrity assertions (blueprint §18). Returns a dict of counts; raises on failure."""
    pop = mb[mb["in_model_population"]]
    assert mb["row_id"].is_unique, "row_id not unique"
    assert pop["race_driver_id"].is_unique, "(race, driver) not unique in model population"
    assert len(mb) == 25873 and len(pop) == 25066
    assert pop["race_id"].nunique() == 1138
    assert mb["race_date"].notna().all() and mb["season"].notna().all()
    winners = pop.groupby("race_id")["target_winner"].sum()
    shared = pop.groupby("race_id")["exclude_from_winner_metrics"].first()
    assert (winners[~shared] == 1).all(), "every non-shared-win race must have exactly one winner"
    assert (winners[shared] == 2).all()
    # targets form 1..k within each race (ties only for shared positions, P-02)
    for rid, g in pop.groupby("race_id"):
        t = g["target_finish_position"]
        assert t.min() == 1 and t.max() <= len(g)
        if not g["shared_position"].any():
            assert sorted(t) == list(range(1, len(g) + 1)), rid
    assert pop["grid_position"].between(1, 40).sum() + pop["grid_position"].isna().sum() == len(pop)
    assert (pop["wx_available"]).all(), "every model race has usable weather (F-006)"
    assert (mb.loc[~mb["in_model_population"], "target_finish_position"].isna()).all()
    return {
        "rows": len(mb), "model_population_rows": len(pop), "model_races": pop["race_id"].nunique(),
        "starters_with_qualifying": int(pop["qualifying_position"].notna().sum()),
        "starters_with_sprint": int(pop["sprint_position"].notna().sum()),
        "grid_unknown_in_population": int(pop["grid_position"].isna().sum()),
        "pit_lane_starts": int((pop["grid_status"] == "pit_lane_start").sum()),
        "dnf_rows": int(pop["target_dnf"].sum()),
    }


def main():
    cfg = load_config()
    interim = PROJECT_ROOT / cfg["paths"]["data_interim"]
    processed = PROJECT_ROOT / cfg["paths"]["data_processed"]
    processed.mkdir(parents=True, exist_ok=True)
    mb = build_master_base(interim)
    stats = check_master_base(mb)
    mb.to_csv(processed / "master_base.csv", index=False)
    for k, v in stats.items():
        print(f"{k:32s} {v}")
    print(f"wrote {processed / 'master_base.csv'}")


if __name__ == "__main__":
    main()
