"""
D-033 model selection: apply the pre-registered rule per model, measure weather sensitivity, install the result.

Inputs: models/algorithms_wx/v1 and models/algorithms_wx/v2 (python -m src.train_algorithms --features v1|v2 --out ...).
Rule (decided before results): use v2 (weather) for a model if its 2017-2019 winner log-loss is lower than v1's AND its
2020-2022 winner log-loss is not worse by more than 0.01; otherwise keep v1. The 2023-2025 test is reported only.
Outputs: outputs/tables/weather_model_comparison.csv, weather_sensitivity.csv; chosen bundles + metadata.json in
models/algorithms (the previous production set is in models/algorithms_d023_backup).

Run:  python -m src.select_weather_models
"""

import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from src import simulate
from src.algorithms import REGISTRY
from src.config import PROJECT_ROOT, load_config
from src.multi_sim import score_table
from src.ranking import _positions

MARGIN = 0.01
CFG = load_config()
MODELS = PROJECT_ROOT / CFG["paths"]["models"]
TABLES = PROJECT_ROOT / CFG["paths"]["tables"]
KEYS = [k for k, s in REGISTRY.items() if s["family"] in ("classification", "regression")]
# one change at a time from the dry median (all other conditions fixed); ranges = the app's slider ends
SCENARIOS = {"wet (0.65 mm/h)": {"weather": "wet"}, "heavy rain (3 mm/h)": {"rain_mm_h": 3.0, "humidity_pct": 90},
             "hot (33 °C)": {"air_temp_c": 33}, "cold (10 °C)": {"air_temp_c": 10},
             "windy (35 km/h)": {"wind_kmh": 35}, "humid (95%)": {"humidity_pct": 95}, "dry air (20%)": {"humidity_pct": 20}}


def expected_positions(b, df, n=4000, seed=42):
    """Mean finishing position over n races from the given (fixed) grid with shared noise; same method as the app."""
    m = len(df)
    noise = np.random.default_rng(seed).gumbel(size=(n, m))
    t = score_table(b, df)
    return _positions(t[np.arange(m), np.arange(m)][None, :] + noise).mean(0)


def sensitivity(bundles):
    rows = []
    for circuit in ("spa", "monza", "monaco", "silverstone"):
        grid = simulate.default_grid(circuit)["entries"]
        base_df = simulate.build_scenario(circuit, grid, {"weather": "dry"})[0]
        for key, b in bundles.items():
            base = expected_positions(b, base_df)
            for name, cond in SCENARIOS.items():
                alt = expected_positions(b, simulate.build_scenario(circuit, grid, cond)[0])
                d = alt - base
                rows.append({"algorithm": key, "feature_set": b["feature_set"], "circuit": circuit, "scenario": name,
                             "mean_abs_change": float(np.abs(d).mean()), "max_abs_change": float(np.abs(d).max()),
                             "podium_changed": list(np.argsort(base)[:3]) != list(np.argsort(alt)[:3])})
    return pd.DataFrame(rows)


def main():
    v = {s: json.loads((MODELS / "algorithms_wx" / s / "metadata.json").read_text())["algorithms"] for s in ("v1", "v2")}
    rows, chosen = [], {}
    for k in KEYS:
        a, b = v["v1"][k], v["v2"][k]
        better_cal = b["calibration_winner_logloss"] < a["calibration_winner_logloss"]
        ok_eval = b["winner_logloss"] <= a["winner_logloss"] + MARGIN
        chosen[k] = "v2" if better_cal and ok_eval else "v1"
        for s, m in (("v1", a), ("v2", b)):
            rows.append({"algorithm": k, "family": REGISTRY[k]["family"], "feature_set": s, "chosen": chosen[k] == s,
                         **{c: m.get(c) for c in m if isinstance(m.get(c), (int, float)) and not isinstance(m.get(c), bool)}})
    comp = pd.DataFrame(rows)
    comp.to_csv(TABLES / "weather_model_comparison.csv", index=False)
    print(comp.pivot(index="algorithm", columns="feature_set", values=["calibration_winner_logloss", "winner_logloss",
                                                                       "test_winner_logloss"]).round(4).to_string())
    print("chosen:", chosen)

    # install: chosen bundles (+ the unchanged Plackett-Luce and grid baseline from the v1 run)
    dst = MODELS / "algorithms"
    meta = json.loads((MODELS / "algorithms_wx" / "v1" / "metadata.json").read_text())
    for k in KEYS:
        shutil.copy2(MODELS / "algorithms_wx" / chosen[k] / f"{k}.joblib", dst / f"{k}.joblib")
        meta["algorithms"][k] = v[chosen[k]][k] | {"feature_set": chosen[k]}
    for k in ("plackett_luce", "grid_baseline", "density"):
        src = MODELS / "algorithms_wx" / "v1" / f"{k}.joblib"
        if src.exists():
            shutil.copy2(src, dst / f"{k}.joblib")
    ranked = sorted(meta["algorithms"].items(), key=lambda kv: kv[1].get("winner_logloss", 9))
    meta |= {"best_validated": ranked[0][0], "feature_set": "per model (D-033)", "weather_rule": __doc__,
             "chosen_feature_sets": chosen, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2, default=lambda x: None if x != x else float(x)))

    bundles = {k: joblib.load(dst / f"{k}.joblib") for k in KEYS}
    sens = sensitivity(bundles)
    sens.to_csv(TABLES / "weather_sensitivity.csv", index=False)
    print(sens.groupby(["algorithm", "feature_set", "scenario"])["mean_abs_change"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
