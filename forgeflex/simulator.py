import numpy as np
import pandas as pd
from .config import SimulationConfig

MACHINE_PROFILES = {
    "CNC_1": {"idle_kw": 3.0, "active_kw": 18.0, "cycle_seconds": 210, "phase": 0},
    "CNC_2": {"idle_kw": 2.5, "active_kw": 15.0, "cycle_seconds": 270, "phase": 35},
    "COMPRESSOR": {"idle_kw": 1.5, "active_kw": 22.0, "cycle_seconds": 360, "phase": 80},
    "PUMP": {"idle_kw": 0.8, "active_kw": 7.5, "cycle_seconds": 480, "phase": 120},
}

def generate_factory_trace(config: SimulationConfig = SimulationConfig()):
    """Return (machine ground truth, aggregate meter) for a synthetic factory only."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    timestamps = pd.date_range("2026-01-01", periods=config.samples,
                               freq=pd.Timedelta(seconds=config.interval_seconds))
    data = {}
    idx = np.arange(config.samples)
    for name, p in MACHINE_PROFILES.items():
        cycle_len = max(1, p["cycle_seconds"] // config.interval_seconds)
        cycle = ((idx + p["phase"]) // cycle_len) % 2
        slow_len = max(1, 45 * 60 // config.interval_seconds)
        slow = ((idx // slow_len + p["phase"] // 40) % 3) != 0
        active = cycle.astype(bool) & slow
        power = np.where(active, p["active_kw"], p["idle_kw"]).astype(float)
        data[name] = np.clip(power + rng.normal(0, 0.15, config.samples), 0, None)
    machine_df = pd.DataFrame({"timestamp": timestamps, **data})
    total = machine_df[list(MACHINE_PROFILES)].sum(axis=1).to_numpy()
    aggregate_df = pd.DataFrame({
        "timestamp": timestamps,
        "aggregate_power_kw": np.clip(total + rng.normal(0, config.noise_std_kw, config.samples), 0, None)
    })
    return machine_df, aggregate_df
