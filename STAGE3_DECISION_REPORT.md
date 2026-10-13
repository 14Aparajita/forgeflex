# ForgeFlex Stage 3 — Integrated Decision Report

## What this adds

Adds `forgeflex.decision_report` and the `decision-report` CLI command. It consolidates already-generated local artifacts into:

- `decision_report.html` — standalone, presentation-friendly report
- `decision_report.json` — detailed machine-readable report with evidence labels and caveats
- `decision_summary.csv` — compact KPI table for further analysis

It does not retrain models, edit existing data, run the scheduler, or overwrite prior outputs. Missing/unreadable inputs are listed in the report instead of silently being replaced with invented values.

## Install from the repository root (PowerShell)

```powershell
Copy-Item .\forgeflex\cli.py .\forgeflex\cli.py.pre_decision_report.bak
Expand-Archive -Path "$HOME\Downloads\forgeflex_stage3_decision_report_v1.zip" -DestinationPath . -Force
python -m pytest -q
python -m forgeflex.cli --help
```

Expected test result for the packaged source snapshot: **33 passed** (31 existing tests plus 2 new report tests). Your local environment may show dependency/environment differences; investigate any failures before proceeding.

## Run the report

Use after the HIPE audit, activity audit, walk-forward, sensor-placement, and scheduling commands have produced their outputs:

```powershell
python -m forgeflex.cli decision-report `
  --output-dir outputs/decision_report_v1 `
  --hipe-audit outputs/hipe/hipe_audit.json `
  --activity-audit outputs/hipe_activity_audit/hipe_activity_audit.json `
  --walkforward-metrics outputs/hipe_walkforward/hipe_walkforward_metrics.csv `
  --sensor-summary outputs/sensor_placement_v1/sensor_placement_summary.json `
  --sensor-recommendations outputs/sensor_placement_v1/sensor_placement_recommendations.csv `
  --scheduling-summary outputs/scheduling_demo_v1/scheduling_summary.json
```

Open the HTML report:

```powershell
Invoke-Item .\outputs\decision_report_v1\decision_report.html
```

Inspect the JSON or CSV:

```powershell
Get-Content .\outputs\decision_report_v1\decision_report.json -Raw
Import-Csv .\outputs\decision_report_v1\decision_summary.csv | Format-Table -AutoSize
```

If an artifact is stored elsewhere, pass its actual path. The command can still generate a partial report when files are missing, but the missing paths will be called out explicitly.

## Evidence and interpretation safeguards

- HIPE is a public historical dataset, not the user's factory data.
- The activity threshold is heuristic, not official machine-state ground truth.
- Sensor-placement ranking is retrospective validation prioritization, not calibrated uncertainty, ROI, or an automatic installation order.
- Scheduling is `SIMULATION_ONLY`; the default jobs and tariff are synthetic/illustrative.
- Lower simulated cost is not the same as lower energy consumption.
- This stage consolidates the modules in one report; it does **not** yet feed inference residuals or sensor rankings into the scheduling optimizer as constraints/objective terms.

## Scope

The archive contains only the complete replacement `forgeflex/cli.py`, new `forgeflex/decision_report.py`, new `tests/test_decision_report.py`, and this guide. It does not contain or overwrite `outputs`, raw data, model artifacts, or the existing module implementations.
