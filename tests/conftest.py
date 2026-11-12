import pandas as pd

from src.config import PROJECT_ROOT


def prime_field(circuit_id):
    """All-time field in prime form at the circuit's capacity: a cross-era fixture for the general simulator tests
    (the app itself races only the 2025 reference pool)."""
    proc = PROJECT_ROOT / "data/processed"
    cap = int(pd.read_csv(proc / "circuit_grid_capacity.csv").set_index("circuit_id").loc[circuit_id, "grid_capacity"])
    p = pd.read_csv(proc / "driver_primes.csv")
    p = p[p["greats_eligible"]].sort_values("prime_rank").head(cap)
    return [{"driver_id": r.driver_id, "season": int(r.prime_season), "constructor_id": r.prime_constructor_id,
             "constructor_season": int(r.prime_season)} for r in p.itertuples()]
