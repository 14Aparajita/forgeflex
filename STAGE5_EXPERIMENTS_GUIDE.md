# ForgeFlex Stage 5 — Scheduling Experiment Suite

## Purpose

This adds a reproducible paired scenario suite around the existing baseline and greedy tariff-aware scheduler. Each scenario compares both methods on the exact same generated jobs and tariff. The suite tests multiple seeds, job loads, and two illustrative tariff conditions.

## Install

Back up the current CLI before extracting this package:

```powershell
Copy-Item .\forgeflex\cli.py .\forgeflex\cli.py.pre_experiments.bak
Expand-Archive -Path "$HOME\Downloads\forgeflex_stage5_experiments_v1.zip" -DestinationPath . -Force
python -m pytest -q
python -m forgeflex.cli --help
```

The ZIP includes a complete CLI replacement based on the Stage 3 CLI, retaining the `decision-report`, `schedule`, sensor-placement, HIPE, inference, and other existing commands. It adds `experiments` and `forgeflex/experiments.py`.

## Run the full default experiment suite

```powershell
python -m forgeflex.cli experiments `
  --output-dir outputs/scheduling_experiments_v1 `
  --seeds 42 7 21 `
  --job-counts 10 20 30 `
  --horizon-slots 24 `
  --slot-hours 1.0
```

This runs 3 seeds × 3 job counts × 2 tariff conditions = 18 paired scenarios. Each scenario includes the baseline and optimized schedule, so 36 schedules are evaluated. It writes:

- `experiment_results.csv` — one row per scenario, with both methods' cost, energy, peak demand, scheduled jobs, deadline misses, and production units.
- `experiment_summary.json` — aggregated outcomes per tariff condition, plus methods, parameters, and limitations.
- `scenario_inputs/` — generated job and tariff inputs.
- `runs/` — complete artifacts for each paired comparison.

## Smoke test first (recommended)

```powershell
python -m forgeflex.cli experiments `
  --output-dir outputs/scheduling_experiments_smoke `
  --seeds 42 `
  --job-counts 10 `
  --horizon-slots 24
```

## Interpretation rules

- All experiment outputs are `SIMULATION_ONLY`.
- The time-of-use tariff and flat tariff are illustrative, not real utility tariffs.
- The current optimizer is a greedy heuristic; this stage does not add a global solver.
- The optimizer may tie or lose on some scenarios. Report the full distribution, including negative improvements; do not report only favorable runs.
- Energy may stay equal while cost changes because the job power and duration are fixed and the schedule shifts between tariff slots.
- Equal production units only reflects unchanged job quantities. Product quality is not modeled.
- No result establishes factory savings, ROI, or deployment readiness.
