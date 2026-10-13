"""Reproducible multi-scenario evaluation for ForgeFlex scheduling.

All scenarios in this module are simulations. It compares the existing
earliest-available baseline with the existing tariff-aware heuristic under
matched jobs and tariff inputs; it does not claim real-factory savings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .scheduling import run_scheduling_simulation


def _flat_tariff(horizon: int, price: float = 7.0) -> pd.DataFrame:
    return pd.DataFrame({"slot": range(horizon), "price_per_kwh": [float(price)] * horizon})


def run_scheduling_experiments(
    output_dir: str | Path,
    seeds: list[int] | None = None,
    job_counts: list[int] | None = None,
    horizon_slots: int = 24,
    slot_hours: float = 1.0,
) -> dict[str, Any]:
    """Run paired simulations across seeds, job loads, and tariff conditions.

    For every pair, baseline and optimized methods receive identical jobs and
    tariffs. Tariff conditions are the simulator's illustrative time-of-use
    tariff and a flat illustrative tariff. No external tariff is implied.
    """
    seeds = [42, 7, 21] if seeds is None else [int(x) for x in seeds]
    job_counts = [10, 20, 30] if job_counts is None else [int(x) for x in job_counts]
    if not seeds or not job_counts:
        raise ValueError("seeds and job_counts must each contain at least one value.")
    if any(x < 1 for x in job_counts):
        raise ValueError("Every job count must be >= 1.")
    if horizon_slots < 8:
        raise ValueError("horizon_slots must be >= 8.")
    if not np.isfinite(slot_hours) or slot_hours <= 0:
        raise ValueError("slot_hours must be a finite positive number.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_index = 0
    for seed in seeds:
        for job_count in job_counts:
            # Create one deterministic job set, then reuse it for both tariff conditions.
            run_index += 1
            job_source_dir = root / "scenario_inputs" / f"seed_{seed}_jobs_{job_count}"
            generated = run_scheduling_simulation(
                output_dir=job_source_dir,
                seed=seed,
                job_count=job_count,
                horizon_slots=horizon_slots,
                slot_hours=slot_hours,
            )
            jobs_path = generated["jobs_csv"]
            tou_path = generated["tariff_csv"]
            flat_path = job_source_dir / "flat_tariff.csv"
            _flat_tariff(horizon_slots).to_csv(flat_path, index=False)

            for tariff_name, tariff_path in (("illustrative_tou", tou_path), ("illustrative_flat", flat_path)):
                scenario_id = f"seed{seed}_jobs{job_count}_{tariff_name}"
                result = run_scheduling_simulation(
                    output_dir=root / "runs" / scenario_id,
                    jobs_csv=jobs_path,
                    tariff_csv=tariff_path,
                    seed=seed,
                    job_count=job_count,
                    horizon_slots=horizon_slots,
                    slot_hours=slot_hours,
                )
                summary = result["summary"]
                base = summary["baseline"]
                opt = summary["optimized"]
                base_cost = float(base["total_energy_cost"])
                opt_cost = float(opt["total_energy_cost"])
                cost_reduction_pct = ((base_cost - opt_cost) / base_cost * 100.0) if base_cost > 0 else None
                rows.append({
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "job_count_requested": job_count,
                    "horizon_slots": horizon_slots,
                    "slot_hours": slot_hours,
                    "tariff_condition": tariff_name,
                    "data_type": "SIMULATION_ONLY",
                    "baseline_jobs_scheduled": base["jobs_scheduled"],
                    "optimized_jobs_scheduled": opt["jobs_scheduled"],
                    "baseline_deadline_misses": base["deadline_misses_among_scheduled_jobs"],
                    "optimized_deadline_misses": opt["deadline_misses_among_scheduled_jobs"],
                    "baseline_energy_kwh": base["total_energy_kwh"],
                    "optimized_energy_kwh": opt["total_energy_kwh"],
                    "baseline_cost": base_cost,
                    "optimized_cost": opt_cost,
                    "cost_reduction_percent": cost_reduction_pct,
                    "baseline_peak_kw": base["peak_aggregate_demand_kw"],
                    "optimized_peak_kw": opt["peak_aggregate_demand_kw"],
                    "peak_change_kw": opt["peak_aggregate_demand_kw"] - base["peak_aggregate_demand_kw"],
                    "baseline_production_units": base["total_production_units_scheduled"],
                    "optimized_production_units": opt["total_production_units_scheduled"],
                    "jobs_csv": str(jobs_path),
                    "tariff_csv": str(tariff_path),
                })

    detail = pd.DataFrame(rows)
    detail_path = root / "experiment_results.csv"
    detail.to_csv(detail_path, index=False)

    summaries = []
    for tariff_name, group in detail.groupby("tariff_condition", sort=True):
        values = pd.to_numeric(group["cost_reduction_percent"], errors="coerce")
        summaries.append({
            "tariff_condition": tariff_name,
            "scenarios": int(len(group)),
            "mean_cost_reduction_percent": float(values.mean()) if values.notna().any() else None,
            "median_cost_reduction_percent": float(values.median()) if values.notna().any() else None,
            "min_cost_reduction_percent": float(values.min()) if values.notna().any() else None,
            "max_cost_reduction_percent": float(values.max()) if values.notna().any() else None,
            "scenarios_with_cost_reduction": int((values > 1e-9).sum()),
            "scenarios_with_cost_increase": int((values < -1e-9).sum()),
            "scenarios_with_equal_cost": int((values.abs() <= 1e-9).sum()),
            "total_baseline_deadline_misses": int(group["baseline_deadline_misses"].sum()),
            "total_optimized_deadline_misses": int(group["optimized_deadline_misses"].sum()),
            "all_jobs_scheduled_both_methods": bool(
                (group["baseline_jobs_scheduled"] == group["job_count_requested"]).all()
                and (group["optimized_jobs_scheduled"] == group["job_count_requested"]).all()
            ),
            "production_units_preserved_all_scenarios": bool(
                (group["baseline_production_units"] == group["optimized_production_units"]).all()
            ),
        })

    summary = {
        "data_type": "SIMULATION_ONLY",
        "title": "ForgeFlex paired multi-scenario scheduling experiment",
        "scenario_count": int(len(detail)),
        "seeds": seeds,
        "job_counts": job_counts,
        "horizon_slots": int(horizon_slots),
        "slot_hours": float(slot_hours),
        "methods": {
            "baseline": "Existing earliest-available schedule with per-machine EDD ordering.",
            "optimized": "Existing greedy tariff-aware schedule with the same per-machine EDD ordering.",
        },
        "comparison_policy": "Within each scenario, both methods use the same generated jobs and the same tariff input.",
        "tariff_conditions": [
            "illustrative_tou: the simulator's synthetic time-of-use tariff; not a real utility tariff.",
            "illustrative_flat: a constant illustrative price of 7 currency units/kWh; not a real tariff.",
        ],
        "limitations": [
            "All jobs and tariffs in this experiment suite are synthetic or illustrative.",
            "The current optimizer is a greedy heuristic, not a globally optimal solver.",
            "Results do not establish real-factory energy savings, cost savings, feasibility, or ROI.",
            "Equal production units in this simulation follow from fixed job quantities; product quality is not modeled.",
            "The same energy per job is assumed across schedules; tariff shifting can reduce cost without reducing kWh.",
            "The experiment suite does not include a MILP/CP-SAT comparator in this version.",
        ],
        "aggregate_by_tariff_condition": summaries,
        "outputs": {"experiment_results_csv": str(detail_path)},
    }
    summary_path = root / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_json": str(summary_path), "results_csv": str(detail_path), "summary": summary}
