"""Verify the frozen data, models and local assets shipped with APEX.

Default: verify without dependencies or network access.
Maintainers: --write records reviewed, Git-indexed artifacts after an intentional update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/ARTIFACT_MANIFEST.json"
PREFIXES = ("data/interim/", "data/processed/", "data/external/", "models/", "frontend/public/")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        artifacts = [{"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": digest(ROOT / name)}
                     for name in sorted(names) if name.startswith(PREFIXES) and not name.endswith(".gitkeep")]
        MANIFEST.write_text(json.dumps({"format": 1, "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {len(artifacts)} reviewed artifacts")
        return
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    failures = []
    for item in artifacts:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise SystemExit("Missing or changed artifacts:\n" + "\n".join(failures))
    print(f"PASS {len(artifacts)} frozen artifacts verified; {sum(x['bytes'] for x in artifacts):,} bytes; no downloads required")


if __name__ == "__main__":
    main()
