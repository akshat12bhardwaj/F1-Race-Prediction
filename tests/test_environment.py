"""
Step 1 environment check.

Confirms that Python and every required package work, that installed versions
match requirements.txt, and that config.yaml loads with the blueprint values.

Run from the project root, either way:
    python -m pytest tests/test_environment.py -v
    python tests/test_environment.py
"""

import importlib
import io
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# import name -> name used in requirements.txt
PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "requests": "requests",
    "yaml": "PyYAML",
    "pytest": "pytest",
    "ipykernel": "ipykernel",
}


def read_pinned_versions():
    """Return {package_name: version} from the '==' lines of requirements.txt."""
    pins = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==")
            pins[name.strip().lower()] = version.strip()
    return pins


def test_python_version():
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, found {sys.version}"


def test_packages_import_and_match_pins():
    from importlib.metadata import version

    pins = read_pinned_versions()
    for import_name, dist_name in PACKAGES.items():
        importlib.import_module(import_name)  # fails if the package is missing
        installed = version(dist_name)
        expected = pins[dist_name.lower()]
        assert installed == expected, f"{dist_name}: installed {installed}, pinned {expected}"


def test_packages_do_basic_work():
    import joblib
    import matplotlib
    matplotlib.use("Agg")  # draw without opening a window
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    df = pd.DataFrame({"x": np.arange(10), "y": [0] * 5 + [1] * 5})
    model = LogisticRegression(random_state=42).fit(df[["x"]], df["y"])
    new_row = pd.DataFrame({"x": [9]})
    assert model.predict(new_row)[0] == 1

    buffer = io.BytesIO()          # save and reload the model in memory
    joblib.dump(model, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    assert restored.predict(new_row)[0] == 1

    fig, ax = plt.subplots()
    ax.plot(df["x"], df["y"])
    plt.close(fig)


def test_requests_is_usable():
    import requests
    session = requests.Session()  # no network call in Step 1
    assert session is not None


def test_jupyter_runs():
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "--version"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "notebook" in result.stdout.lower()


def test_config_matches_blueprint():
    import yaml

    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    split = config["split"]

    assert config["random_seed"] == 42
    assert config["seasons"]["start"] == 1950
    assert config["seasons"]["end"] == 2025
    assert split["validation_years"] == [2022]
    assert split["test_years"] == [2023, 2024, 2025]
    # train < validation < test, with no overlap
    assert split["train_end"] < min(split["validation_years"])
    assert max(split["validation_years"]) < min(split["test_years"])
    assert config["api"]["jolpica"]["base_url"].startswith("https://api.jolpi.ca")


def test_folders_exist():
    for folder in ["data/raw", "data/interim", "data/processed", "notebooks",
                   "src", "tests", "models", "outputs/figures", "outputs/tables", "docs"]:
        assert (PROJECT_ROOT / folder).is_dir(), f"Missing folder: {folder}"


if __name__ == "__main__":
    # Plain-script mode: run each check and print PASS/FAIL.
    checks = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS  {check.__name__}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {check.__name__}: {error}")
    print(f"\nPython {sys.version.split()[0]} at {sys.executable}")
    print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
    sys.exit(1 if failures else 0)
