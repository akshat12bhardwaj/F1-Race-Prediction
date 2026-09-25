"""
Project configuration helpers.

PROJECT_ROOT is the folder that contains config.yaml, found from this file's
location, so code works no matter which folder it is started from.
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Read config.yaml (or another YAML file) and return it as a dictionary."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
