# ForgeFlex Stage 6 — Conservative Constraint-Aware Scheduling

## Purpose
Adds `constraint-aware-schedule`, a simulation-only conservative local-search method. It starts from the existing earliest-available baseline and moves only jobs that met their baseline deadline into lower-cost contiguous slots that still meet their deadlines and do not overlap another job on the same machine. Baseline deadline misses and unscheduled jobs remain unchanged. It is not a global optimizer and does not guarantee global cost optimality.

## Install
From the ForgeFlex project root, back up the CLI, then expand this ZIP into the project root. The package replaces `forgeflex/cli.py` and adds `forgeflex/constraint_aware.py` and `tests/test_constraint_aware.py`. It does not contain project data or outputs.

```powershell
Copy-Item .\forgeflex\cli.py .\forgeflex\cli.py.pre_constraint_aware.bak
Expand-Archive -Path "$HOME\Downloads\forgeflex_stage6_constraint_aware_v1.zip" -DestinationPath . -Force
python -m pytest -q
python -m forgeflex.cli --help
```

## Smoke run
```powershell
python -m forgeflex.cli constraint-aware-schedule `
  --output-dir outputs/constraint_aware_smoke `
  --seed 42 --job-count 20 --horizon-slots 24 --slot-hours 1.0
Get-Content .\outputs\constraint_aware_smoke\constraint_aware_summary.json -Raw
```

## Existing 18-scenario inputs
For a previously generated scenario, use its exact jobs and tariff CSVs to reproduce a paired comparison. Example:
```powershell
python -m forgeflex.cli constraint-aware-schedule `
  --jobs outputs/scheduling_experiments_v1/scenario_inputs/seed_7_jobs_30/scenario_jobs.csv `
  --tariff outputs/scheduling_experiments_v1/scenario_inputs/seed_7_jobs_30/scenario_tariff.csv `
  --output-dir outputs/constraint_aware_seed7_jobs30 `
  --horizon-slots 24 --slot-hours 1.0
```

Note: that example uses the generated time-of-use tariff. To compare against the flat tariff, point `--tariff` to that scenario directory's `flat_tariff.csv` instead.

## Outputs
- `schedule_baseline.csv`
- `schedule_constraint_aware.csv`
- `schedule_slot_profiles.csv`
- `scenario_jobs.csv`
- `scenario_tariff.csv`
- `constraint_aware_summary.json`

## Guardrails
All default jobs/tariffs are synthetic or illustrative. Do not present output as factory savings, verified feasibility in a real plant, or ROI. This heuristic preserves baseline deadline outcomes and scheduled-job count; it does not repair infeasibility already present in the baseline, model setup/precedence/labor/quality constraints, or guarantee global optimality.
