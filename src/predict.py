"""
Phase 5: prediction engine and CLI (blueprint §30).

A race in season S is predicted by the walk-forward pipelines trained only on seasons < S
(models/walkforward/S/). Reserved test seasons are LOCKED until the approved one-time
final evaluation has been run (metadata `test_evaluated`); then models/final/ is used.

Inputs used for inference are ONLY the T0 feature columns. Actual outcomes are held in a
separate table and returned only by `actual_results` (reveal-after-prediction).

The whole race field is always predicted; a driver subset only filters what is displayed,
so probabilities stay valid for the real race.

CLI:  python -m src.predict --season 2022 --round 5 [--drivers leclerc,sainz] [--reveal]
"""

import argparse
import json
import sys
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluate import final_order, normalize_podium, normalize_win
from src.features import FEATURE_GROUPS, FEATURES, TARGETS, load_features

GROUP_LABELS = {"grid_quali": "Grid & qualifying", "sprint": "Sprint result", "driver_form": "Driver recent form",
                "driver_career": "Driver career record", "championship": "Championship standing",
                "constructor": "Constructor form", "circuit": "Circuit history",
                "weather_prestart": "Pre-start weather"}
INFO_COLS = ["race_driver_id", "race_id", "season", "round", "race_date", "race_name", "circuit_id",
             "circuit_name", "country", "driver_id", "driver_code", "driver_first_name", "driver_last_name",
             "driver_number", "constructor_id", "constructor_name"]


# Pipelines trained before 2021 never saw a sprint, so their imputer drops the all-empty
# sprint_position column (consistently at fit and predict). Benign; silence only that message.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


class RaceUnavailable(Exception):
    """The race cannot be predicted (unknown, locked, or outside the out-of-sample seasons)."""


class ModelsMissing(Exception):
    pass


def assemble_order(df, podium_method):
    """Unique deterministic order: the predicted podium (top 3 by the E4-selected method) ordered
    by win probability, followed by everyone else in regression order. Ties -> grid -> driver_id."""
    score = {"regression": df["pred_position"], "podium_classifier": -df["podium_prob"],
             "winner_classifier": -df["win_prob"]}[podium_method]
    base = final_order(df.assign(score=score), "score")
    reg = final_order(df.assign(score=df["pred_position"]), "score")
    d = df.assign(_rest=(base > 3).astype(int), _key=np.where(base <= 3, -df["win_prob"], reg))
    order = d.sort_values(["race_id", "_rest", "_key", "grid_position", "driver_id"], na_position="last")
    return order.groupby("race_id").cumcount().add(1).reindex(df.index)


class Predictor:
    def __init__(self, models_dir=None, features=None):
        cfg = load_config()
        self.models_dir = models_dir or PROJECT_ROOT / cfg["paths"]["models"]
        meta_path = self.models_dir / "metadata_v1.json"
        if not meta_path.exists():
            raise ModelsMissing(f"{meta_path} not found; run python -m src.train")
        self.meta = json.loads(meta_path.read_text())
        assert self.meta["features"] == FEATURES, "feature list changed since training"
        mf = features if features is not None else load_features()
        # inputs and outcomes are kept apart; inference never reads self._results
        self._inputs = mf[INFO_COLS + FEATURES]
        self._results = mf[["race_id", "driver_id"] + TARGETS]
        self.test_seasons = set(cfg["split"]["test_years"])
        self.replay_seasons = set(self.meta["walk_forward_seasons"])
        if self.meta.get("test_evaluated"):
            self.replay_seasons |= self.test_seasons

    # ---------- catalogue ----------
    def season_status(self, season):
        if season in self.replay_seasons:
            return "available"
        if season in self.test_seasons:
            return "locked_test"
        return "retrospective"

    def model_info(self, season):
        """Which pipelines predict this season, and whether that is out-of-sample."""
        first = min(self.meta["walk_forward_seasons"])
        if self.season_status(season) == "retrospective":
            return {"mode": "retrospective", "trained_on": f"{self.meta['train_start']}-{first - 1}",
                    "note": f"Retrospective fit: the model (trained {self.meta['train_start']}-{first - 1}) saw this "
                            "season's results during training, so this is NOT a genuine prediction."}
        if season in self.test_seasons:
            return {"mode": "out_of_sample", "trained_on": self.meta["final_models_trained_on"],
                    "note": "Out-of-sample: models trained only on seasons up to 2022 (one-time test seasons)."}
        return {"mode": "out_of_sample", "trained_on": f"{self.meta['train_start']}-{season - 1}",
                "note": "Out-of-sample: models trained only on seasons before this one (walk-forward)."}

    def seasons(self):
        s = sorted(self._inputs["season"].unique())
        return [{"season": int(x), "status": self.season_status(int(x))} for x in s]

    def races(self, season):
        r = self._inputs[self._inputs["season"] == season]
        g = r.groupby("race_id", sort=False).agg(
            round=("round", "first"), race_name=("race_name", "first"), race_date=("race_date", "first"),
            circuit_id=("circuit_id", "first"), circuit_name=("circuit_name", "first"),
            country=("country", "first"), starters=("driver_id", "size")).reset_index()
        status = self.season_status(season)
        g["status"] = status
        g["race_date"] = pd.to_datetime(g["race_date"]).dt.strftime("%Y-%m-%d")
        return g.sort_values("round").to_dict("records")

    def _race_rows(self, season, rnd):
        r = self._inputs[(self._inputs["season"] == season) & (self._inputs["round"] == rnd)]
        if r.empty:
            raise RaceUnavailable(f"no model-eligible race {season} round {rnd}")
        return r

    def drivers(self, season, rnd):
        r = self._race_rows(season, rnd).sort_values(["grid_position", "driver_id"], na_position="last")
        cols = ["driver_id", "driver_code", "driver_first_name", "driver_last_name", "driver_number",
                "constructor_id", "constructor_name", "grid_position", "qualifying_position"]
        return r[cols].replace({np.nan: None}).to_dict("records")

    # ---------- models ----------
    @lru_cache(maxsize=16)
    def _models(self, season):
        status = self.season_status(season)
        if status == "locked_test":
            raise RaceUnavailable(f"{season} is a reserved test season (locked until the approved "
                                  "final evaluation)")
        if status == "retrospective":
            d = self.models_dir / f"walkforward/{min(self.meta['walk_forward_seasons'])}"
        else:
            d = self.models_dir / ("final" if season in self.test_seasons else f"walkforward/{season}")
        try:
            models = {t: joblib.load(d / f"{t}.joblib") for t in ("winner", "podium", "position")}
        except FileNotFoundError as e:
            raise ModelsMissing(str(e)) from e
        for m in models.values():   # parallel tree averaging sums in varying order -> 1-ulp jitter
            est = m.steps[-1][1]
            if hasattr(est, "n_jobs"):
                est.set_params(n_jobs=1)
        return models

    # ---------- prediction ----------
    def predict_race(self, season, rnd, explain=True):
        season, rnd = int(season), int(rnd)
        m = self._models(season)
        # pre-race order only (the features file is sorted by actual finish)
        rows = self._race_rows(season, rnd).sort_values(["grid_position", "driver_id"], na_position="last") \
            .reset_index(drop=True)
        X = rows[FEATURES]
        out = rows[[c for c in INFO_COLS if c in rows]].copy()
        out["grid_position"] = rows["grid_position"]
        out["qualifying_position"] = rows["qualifying_position"]
        out["raw_win_score"] = m["winner"].predict_proba(X)[:, 1]
        out["win_prob"] = normalize_win(out["raw_win_score"], out["race_id"], self.meta["tau"])
        out["podium_prob"] = normalize_podium(m["podium"].predict_proba(X)[:, 1], out["race_id"])
        out["pred_position"] = m["position"].predict(X)
        out["pred_rank"] = assemble_order(out, self.meta["podium_method"])
        if explain:
            out["factors"] = self._explain(m["position"], X)
        out["key_inputs"] = rows[["driver_last5_avg_finish", "constructor_last5_avg_finish",
                                  "driver_champ_pos_before_race", "driver_career_starts",
                                  "quali_gap_to_pole_pct"]].replace({np.nan: None}).to_dict("records")
        return out.sort_values("pred_rank").reset_index(drop=True)

    @staticmethod
    def _explain(model, X):
        """Per-driver group occlusion on the position model: set a feature group to the race
        median and measure the change in predicted position. effect > 0 = the group's values
        move this driver UP (towards P1) relative to a median driver in this race."""
        base = model.predict(X)
        med = X.median(numeric_only=True)
        effects = {}
        for g, cols in FEATURE_GROUPS.items():
            Xg = X.assign(**{c: med[c] for c in cols})
            effects[g] = model.predict(Xg) - base   # positive: without this group, worse position
        res = []
        for i in range(len(X)):
            f = sorted(({"group": g, "label": GROUP_LABELS[g], "effect": round(float(effects[g][i]), 2)}
                        for g in FEATURE_GROUPS), key=lambda d: -abs(d["effect"]))
            res.append(f)
        return res

    # ---------- reveal (outcomes; never used by predict_race) ----------
    def actual_results(self, season, rnd):
        self._models(int(season))   # same availability rules: locked seasons stay locked
        rows = self._race_rows(int(season), int(rnd))
        res = self._results[self._results["race_id"] == rows["race_id"].iloc[0]]
        return res[["driver_id", "target_finish_position", "official_position", "position_text", "status",
                    "points"]].sort_values("target_finish_position")


def race_summary(pred, actual):
    """Prediction vs actual for one race (used by CLI and API)."""
    from scipy.stats import spearmanr
    d = pred.merge(actual, on="driver_id", how="left")
    act_pod = list(actual.sort_values("target_finish_position")["driver_id"][:3])
    pred_pod = list(pred.sort_values("pred_rank")["driver_id"][:3])
    winners = set(actual.loc[actual["target_finish_position"] == 1, "driver_id"])
    return {
        "predicted_podium": pred_pod, "actual_podium": act_pod,
        "winner_correct": pred_pod[0] in winners,
        "podium_overlap": len(set(pred_pod) & set(act_pod)),
        "exact_podium": pred_pod == act_pod,
        "spearman": float(spearmanr(d["pred_rank"], d["target_finish_position"]).statistic),
        "position_mae": float(np.mean(np.abs(d["pred_rank"] - d["target_finish_position"]))),
        "actual_winner_win_prob": float(d.loc[d["driver_id"].isin(winners), "win_prob"].max()),
    }


def main():
    ap = argparse.ArgumentParser(description="Predict a historical F1 race at T0 (grid final, pre-start).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--drivers", help="comma-separated driver ids to display (full field is still modelled)")
    ap.add_argument("--reveal", action="store_true", help="also show the actual result")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    p = Predictor()
    try:
        pred = p.predict_race(a.season, a.round)
    except RaceUnavailable as e:
        raise SystemExit(f"Unavailable: {e}")
    r = pred.iloc[0]
    print("=" * 74)
    print(f" {r['season']} {r['race_name']} · {r['circuit_name']} · {pd.Timestamp(r['race_date']).date()}")
    print(f" Prediction point: final starting grid (T0). Models trained on seasons < {a.season}.")
    print("=" * 74)
    show = pred if not a.drivers else pred[pred["driver_id"].isin(a.drivers.split(","))]
    print(f" {'Pos':>3}  {'Driver':22s} {'Team':20s} {'Grid':>4} {'Win %':>6} {'Podium %':>8}")
    for _, d in show.iterrows():
        grid = "-" if pd.isna(d["grid_position"]) else int(d["grid_position"])
        print(f" {int(d['pred_rank']):>3}  {d['driver_first_name'] + ' ' + d['driver_last_name']:22.22s} "
              f"{d['constructor_name']:20.20s} {grid:>4} {100 * d['win_prob']:6.1f} {100 * d['podium_prob']:8.1f}")
    pod = pred.nsmallest(3, "pred_rank")
    print("-" * 74)
    print(" Predicted podium: " + " · ".join(f"P{i + 1} {x}" for i, x in enumerate(pod["driver_last_name"])))
    if a.reveal:
        s = race_summary(pred, p.actual_results(a.season, a.round))
        print(f" Actual podium:    {' · '.join(s['actual_podium'])}")
        print(f" Winner correct: {s['winner_correct']} · podium overlap {s['podium_overlap']}/3 · "
              f"Spearman {s['spearman']:.2f} · position MAE {s['position_mae']:.2f}")
    print(" Probabilistic estimate from historical patterns; it cannot foresee crashes, failures,")
    print(" safety cars or strategy calls.")


if __name__ == "__main__":
    main()
