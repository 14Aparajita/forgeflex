"""Conservative tariff-aware rescheduling that never worsens baseline deadline outcomes.

Simulation only. Starts from the existing baseline schedule, then moves only jobs
that met their baseline deadline into cheaper, non-overlapping slots that still
meet their deadlines. Jobs already missing deadlines remain unchanged. This is a
local-improvement heuristic, not a global optimizer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .scheduling import _load_jobs, _load_tariff, _schedule, _profile, _summary, run_scheduling_simulation


def run_constraint_aware_schedule(
    output_dir: str | Path,
    jobs_csv: str | Path | None = None,
    tariff_csv: str | Path | None = None,
    seed: int = 42,
    job_count: int = 20,
    horizon_slots: int = 24,
    slot_hours: float = 1.0,
) -> dict[str, Any]:
    if horizon_slots < 1:
        raise ValueError("horizon_slots must be >= 1.")
    if not np.isfinite(slot_hours) or slot_hours <= 0:
        raise ValueError("slot_hours must be a finite positive number.")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Reuse the existing input generation/validation logic and preserve generated inputs.
    if jobs_csv is None or tariff_csv is None:
        generated_dir = outdir / "_generated_inputs"
        generated = run_scheduling_simulation(
            output_dir=generated_dir, jobs_csv=jobs_csv, tariff_csv=tariff_csv,
            seed=seed, job_count=job_count, horizon_slots=horizon_slots, slot_hours=slot_hours,
        )
        jobs_csv = generated["jobs_csv"]
        tariff_csv = generated["tariff_csv"]
    jobs, jobs_source = _load_jobs(jobs_csv, seed, job_count, horizon_slots)
    tariff, tariff_source = _load_tariff(tariff_csv, horizon_slots)
    prices = tariff.set_index("slot")["price_per_kwh"].to_dict()

    baseline = _schedule(jobs, tariff, horizon_slots, "baseline")
    conservative = baseline.copy(deep=True)
    conservative["schedule_mode"] = "constraint_aware"
    conservative["schedule_note"] = conservative["schedule_note"].astype(str)

    # Per-machine local search. Preserve baseline-unscheduled jobs and baseline
    # deadline misses. Only relocate jobs that met their baseline deadlines.
    for machine, group in jobs.groupby("machine", sort=True):
        ids = group["job_id"].astype(str).tolist()
        row_idx = {str(v): i for i, v in conservative.iterrows() if str(conservative.loc[i, "machine"]) == str(machine)}
        machine_rows = [row_idx[jid] for jid in ids if jid in row_idx]
        occupied = np.zeros(horizon_slots, dtype=bool)
        for i in machine_rows:
            row = conservative.loc[i]
            if bool(row["scheduled"]):
                s, e = int(row["start_slot"]), int(row["completion_slot"])
                occupied[s:e] = True

        # Prioritize longer jobs first: moving them has fewer candidate slots and
        # avoids a later short job consuming the only useful window.
        movable = []
        for i in machine_rows:
            row = conservative.loc[i]
            if bool(row["scheduled"]) and bool(row["deadline_met"]):
                movable.append((int(row["duration_slots"]), str(row["job_id"]), i))
        movable.sort(key=lambda x: (-x[0], x[1]))
        for duration, job_id, i in movable:
            row = conservative.loc[i]
            current_start = int(row["start_slot"])
            current_end = int(row["completion_slot"])
            release, due = int(row["release_slot"]), int(row["due_slot"])
            power = float(row["power_kw"])
            # Temporarily release the job's current block while considering moves.
            occupied[current_start:current_end] = False
            current_cost = power * sum(float(prices[k]) for k in range(current_start, current_end))
            candidates = []
            latest = min(due - duration, horizon_slots - duration)
            for start in range(max(0, release), latest + 1):
                end = start + duration
                if not occupied[start:end].any():
                    cost = power * sum(float(prices[k]) for k in range(start, end))
                    candidates.append((cost, start))
            best_cost, best_start = min(candidates, default=(current_cost, current_start), key=lambda x: (x[0], x[1]))
            if best_cost + 1e-9 < current_cost:
                start = int(best_start)
                end = start + duration
                conservative.loc[i, "start_slot"] = start
                conservative.loc[i, "completion_slot"] = end
                conservative.loc[i, "energy_cost"] = float(best_cost) * slot_hours
                conservative.loc[i, "schedule_note"] = "rescheduled_lower_cost_deadline_preserved"
            else:
                start, end = current_start, current_end
                conservative.loc[i, "schedule_note"] = "kept_baseline_slot_no_safe_cost_improvement"
            occupied[start:end] = True

    for frame in (baseline, conservative):
        frame["energy_kwh"] = pd.to_numeric(frame["energy_kwh"], errors="coerce") * slot_hours
        # Recalculate cost consistently from tariff and slot-hours.
        for idx, row in frame.iterrows():
            if bool(row["scheduled"]):
                s, e = int(row["start_slot"]), int(row["completion_slot"])
                frame.loc[idx, "energy_cost"] = float(row["power_kw"]) * sum(float(prices[k]) for k in range(s, e)) * slot_hours

    # Verify core invariants before writing outputs.
    for machine, group in conservative[conservative["scheduled"].astype(bool)].groupby("machine"):
        intervals = sorted((int(r.start_slot), int(r.completion_slot), str(r.job_id)) for r in group.itertuples())
        for left, right in zip(intervals, intervals[1:]):
            if left[1] > right[0]:
                raise RuntimeError(f"Non-overlap invariant failed for machine {machine}: {left[2]} and {right[2]}.")
    baseline_misses = int((baseline[baseline["scheduled"].astype(bool)]["deadline_met"] == False).sum())
    new_misses = int((conservative[conservative["scheduled"].astype(bool)]["deadline_met"] == False).sum())
    if new_misses > baseline_misses:
        raise RuntimeError("Deadline invariant failed: constraint-aware schedule worsened baseline misses.")
    if int(conservative["scheduled"].astype(bool).sum()) != int(baseline["scheduled"].astype(bool).sum()):
        raise RuntimeError("Scheduled-job count changed relative to baseline.")

    base_profile = _profile(baseline, tariff, horizon_slots, slot_hours)
    opt_profile = _profile(conservative, tariff, horizon_slots, slot_hours)
    base_summary = _summary(baseline, base_profile, horizon_slots, slot_hours)
    opt_summary = _summary(conservative, opt_profile, horizon_slots, slot_hours)
    baseline_path = outdir / "schedule_baseline.csv"
    optimized_path = outdir / "schedule_constraint_aware.csv"
    jobs_path = outdir / "scenario_jobs.csv"
    tariff_path = outdir / "scenario_tariff.csv"
    profile_path = outdir / "schedule_slot_profiles.csv"
    summary_path = outdir / "constraint_aware_summary.json"
    jobs.to_csv(jobs_path, index=False); tariff.to_csv(tariff_path, index=False)
    baseline.to_csv(baseline_path, index=False); conservative.to_csv(optimized_path, index=False)
    base_profile.merge(opt_profile, on=["slot", "price_per_kwh"], suffixes=("_baseline", "_constraint_aware")).to_csv(profile_path, index=False)
    base_cost, opt_cost = float(base_summary["total_energy_cost"]), float(opt_summary["total_energy_cost"])
    summary: dict[str, Any] = {
        "data_type": "SIMULATION_ONLY",
        "title": "ForgeFlex conservative constraint-aware scheduling comparison",
        "jobs_input_source": jobs_source, "tariff_input_source": tariff_source,
        "method": "Start from earliest-available baseline; move only baseline deadline-met jobs to cheaper non-overlapping windows that still meet their deadlines. Keep baseline-missed and unscheduled jobs unchanged.",
        "guarantees_checked": {
            "scheduled_job_count_preserved": int(conservative["scheduled"].astype(bool).sum()) == int(baseline["scheduled"].astype(bool).sum()),
            "deadline_misses_not_worse_than_baseline": new_misses <= baseline_misses,
            "per_machine_non_overlap": True,
            "production_units_scheduled_preserved": float(opt_summary["total_production_units_scheduled"]) == float(base_summary["total_production_units_scheduled"]),
        },
        "baseline": base_summary, "constraint_aware": opt_summary,
        "cost_reduction_percent_vs_baseline": ((base_cost - opt_cost) / base_cost * 100.0) if base_cost > 0 else None,
        "limitations": [
            "Simulation only; synthetic jobs and/or illustrative tariff unless supplied otherwise.",
            "Conservative local-improvement heuristic, not a globally optimal solver.",
            "It preserves baseline deadline outcomes but may leave savings unrealized; it does not repair baseline infeasibility.",
            "No setup times, precedence, labor, failures, demand charges, or quality model.",
            "No real-factory savings or ROI is established.",
        ],
        "outputs": {"baseline_schedule_csv": str(baseline_path), "constraint_aware_schedule_csv": str(optimized_path), "jobs_csv": str(jobs_path), "tariff_csv": str(tariff_path), "slot_profiles_csv": str(profile_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_json": str(summary_path), "summary": summary, **summary["outputs"]}
