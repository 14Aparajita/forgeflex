import numpy as np
import pandas as pd

def aggregate_summary(aggregate_df: pd.DataFrame, interval_seconds: int) -> dict:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    power = pd.to_numeric(aggregate_df["aggregate_power_kw"], errors="coerce").to_numpy(dtype=float)
    if len(power) == 0 or not np.isfinite(power).all():
        raise ValueError("power trace must be non-empty and finite")
    return {
        "samples": int(len(power)),
        "energy_kwh": float(power.sum() * interval_seconds / 3600.0),
        "mean_power_kw": float(power.mean()),
        "peak_power_kw": float(power.max()),
        "p95_power_kw": float(np.percentile(power, 95)),
    }

def percentage_reduction(baseline: float, proposed: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline must be > 0")
    return 100.0 * (baseline - proposed) / baseline
