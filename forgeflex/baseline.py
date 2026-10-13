import json
from pathlib import Path
from .config import SimulationConfig
from .metrics import aggregate_summary
from .preprocessing import clean_aggregate
from .simulator import generate_factory_trace

def run_synthetic_baseline(output_dir, config: SimulationConfig) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    machine_df, aggregate_df = generate_factory_trace(config)
    aggregate_df = clean_aggregate(aggregate_df)
    machine_path, aggregate_path = out / "machine_power.csv", out / "aggregate_power.csv"
    machine_df.to_csv(machine_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)
    result = {
        "data_type": "SIMULATED",
        "warning": "Synthetic trace only; not a real factory measurement or savings result.",
        "seed": config.seed,
        "interval_seconds": config.interval_seconds,
        "kpis": aggregate_summary(aggregate_df, config.interval_seconds),
        "files": {"machine_ground_truth": str(machine_path), "aggregate_meter_input": str(aggregate_path)}
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
