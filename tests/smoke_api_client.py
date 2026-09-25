"""
Step 3 live smoke test for the cached Jolpica client.

Makes at most 3 real API requests (fewer if the cache already has them):
  - 2024 Round 1 race results
  - 2024 Round 1 qualifying
  - 2024 Round 5 sprint (Chinese GP; 2024 sprint rounds were found in Step 2)
Then repeats the same 3 calls and checks that none of them used the network.

Cache files go to the real project cache: data/raw/jolpica/

Run from the project root:
    python tests/smoke_api_client.py
(Named smoke_*.py, not test_*.py, so pytest does not run it automatically.)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow "import src" from tests/

from src.api_client import JolpicaClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

REQUESTS = [
    ("2024 R1 race results", "get_race_results", 2024, 1, "Results"),
    ("2024 R1 qualifying", "get_qualifying", 2024, 1, "QualifyingResults"),
    ("2024 R5 sprint", "get_sprint_results", 2024, 5, "SprintResults"),
]


def run_all(client):
    summary = []
    for label, method, season, round_, record_key in REQUESTS:
        races = getattr(client, method)(season, round_)
        assert len(races) == 1, f"{label}: expected 1 race, got {len(races)}"
        records = races[0][record_key]
        assert records, f"{label}: no records returned"
        assert isinstance(records[0]["position"], str), "values must stay strings"
        summary.append((label, races[0]["raceName"], len(records),
                        records[0]["Driver"]["driverId"]))
    return summary


client = JolpicaClient()

print("\n=== Pass 1 (network on first ever run, cache afterwards) ===")
for row in run_all(client):
    print("  %-22s %-22s records=%-3d P1=%s" % row)
network_after_pass1 = client.stats["network_requests"]

print("\n=== Pass 2 (must come entirely from the cache) ===")
run_all(client)
network_in_pass2 = client.stats["network_requests"] - network_after_pass1

print("\nCache files:")
for label, method, season, round_, _ in REQUESTS:
    endpoint = {"get_race_results": "results", "get_qualifying": "qualifying",
                "get_sprint_results": "sprint"}[method]
    path = client.cache_path(f"/{season}/{round_}/{endpoint}.json", {"limit": 100, "offset": 0})
    print(f"  {'OK ' if path.exists() else 'MISSING'} {path.relative_to(client.cache_dir.parents[2]).as_posix()}")

print("\nStats:", client.stats)
print("Network requests in pass 1:", network_after_pass1)
print("Network requests in pass 2:", network_in_pass2)
assert network_in_pass2 == 0, "Pass 2 should not touch the network"
print("\nSMOKE TEST PASSED")
