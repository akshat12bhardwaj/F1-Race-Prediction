"""
Season-based chronological splits (blueprint §27). Years come from config.yaml.

The reserved test seasons (2023-2025) are LOCKED: `development_rows` drops them, and
`season_split` returns test rows only for the split test (T5) and the approved one-time
final evaluation. Nothing in training/selection code reads test outcomes.
"""

import numpy as np

from src.config import load_config


def _split_cfg():
    return load_config()["split"]


def test_seasons():
    return list(_split_cfg()["test_years"])


def development_rows(df):
    """All rows outside the reserved test seasons."""
    return df[~df["season"].isin(test_seasons())]


def season_split(df):
    s = _split_cfg()
    tr = df[(df["season"] >= s["train_start"]) & (df["season"] <= s["train_end"])]
    va = df[df["season"].isin(s["validation_years"])]
    te = df[df["season"].isin(s["test_years"])]
    return tr, va, te


def walk_forward_seasons():
    """Out-of-sample seasons: the CV folds plus the hold-out validation season."""
    s = _split_cfg()
    return list(s["cv_validation_years"]) + list(s["validation_years"])


def expanding_folds(df, val_seasons, train_start=None):
    """Yield (train_idx, val_idx) positional indices: train = seasons < v, validate = season v."""
    start = train_start if train_start is not None else _split_cfg()["train_start"]
    season = df["season"].to_numpy()
    assert not np.isin(season, test_seasons()).any(), "test seasons must not enter CV"
    for v in val_seasons:
        tr = np.where((season >= start) & (season < v))[0]
        va = np.where(season == v)[0]
        if len(va):
            yield tr, va
