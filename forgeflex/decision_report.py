"""Build a consolidated, provenance-aware ForgeFlex decision report.

This module reads existing pipeline artifacts. It does not retrain models, mutate
source data, or imply that public-dataset diagnostics are measurements of a user's
factory. Missing inputs are reported explicitly so the report can be partial.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: str | Path | None, label: str, missing: list[str]) -> dict[str, Any] | None:
    if path is None:
        missing.append(f"{label}: no path supplied")
        return None
    p = Path(path)
    if not p.exists():
        missing.append(f"{label}: file not found ({p})")
        return None
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        missing.append(f"{label}: could not read valid JSON ({exc})")
        return None
    if not isinstance(value, dict):
        missing.append(f"{label}: expected a JSON object")
        return None
    return value


def _read_csv(path: str | Path | None, label: str, missing: list[str]) -> pd.DataFrame | None:
    if path is None:
        missing.append(f"{label}: no path supplied")
        return None
    p = Path(path)
    if not p.exists():
        missing.append(f"{label}: file not found ({p})")
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:  # malformed local artifacts should not destroy other report sections
        missing.append(f"{label}: could not read CSV ({exc})")
        return None


def _weighted_metric(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if value_col not in frame or weight_col not in frame:
        return None
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _walkforward_summary(metrics: pd.DataFrame | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    required = {"model", "status", "mae_kw", "test_rows"}
    if not required.issubset(metrics.columns):
        return {"status": "unavailable", "reason": "metrics CSV lacks required columns", "columns": list(metrics.columns)}
    evaluated = metrics[metrics["status"].astype(str).eq("evaluated")].copy()
    rows = []
    for model, group in evaluated.groupby("model", sort=True):
        item: dict[str, Any] = {
            "model": str(model),
            "evaluated_rows": int(len(group)),
            "weighted_mae_kw": _weighted_metric(group, "mae_kw", "test_rows"),
        }
        if "active_only_mae_kw" in group and "n_active_test" in group:
            item["active_only_mae_kw_weighted_by_active_samples"] = _weighted_metric(
                group, "active_only_mae_kw", "n_active_test"
            )
        rows.append(item)
    return {"status": "available", "evaluated_metric_rows": int(len(evaluated)), "by_model": rows,
            "interpretation": "Weighted summary of saved chronological test-fold metrics; not a guarantee of deployment performance."}


def _num(value: Any, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _esc(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


def _html_report(report: dict[str, Any]) -> str:
    hipe = report.get("hipe") or {}
    activity = report.get("activity_audit") or {}
    wf = report.get("walkforward") or {}
    sensor = report.get("sensor_placement") or {}
    sched = report.get("scheduling") or {}
    sched_base = sched.get("baseline") or {}
    sched_opt = sched.get("optimized") or {}
    selected = sensor.get("selected_for_submeter_validation", []) or []
    selected_html = "".join(f"<li>{_esc(x)}</li>" for x in selected) or "<li>No candidates available</li>"
    wf_rows = ""
    for row in wf.get("by_model", []) if isinstance(wf, dict) else []:
        wf_rows += ("<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _esc(row.get("model")), _num(row.get("weighted_mae_kw")),
            _num(row.get("active_only_mae_kw_weighted_by_active_samples"))))
    if not wf_rows:
        wf_rows = '<tr><td colspan="3">Walk-forward metrics unavailable or unreadable</td></tr>'
    warnings = report.get("missing_or_unreadable_inputs", [])
    warning_html = "".join(f"<li>{_esc(w)}</li>" for w in warnings) or "<li>All configured input artifacts were read.</li>"
    caveats = report.get("interpretation_guardrails", [])
    caveat_html = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    cost_delta = (sched.get("delta_optimized_minus_baseline") or {}).get("energy_cost")
    cost_pct = (sched.get("percent_change_optimized_vs_baseline") or {}).get("energy_cost")
    peak_delta = (sched.get("delta_optimized_minus_baseline") or {}).get("peak_aggregate_demand_kw")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForgeFlex Decision Report</title><style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172338;--muted:#64748b;--line:#dbe3ee;--accent:#176b87;--warn:#fff4d6}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
header{{background:#102c43;color:white;padding:28px max(20px,calc((100vw - 1120px)/2))}}header h1{{margin:0 0 4px;font-size:28px}}header p{{margin:0;color:#d4e5ef}}main{{max-width:1120px;margin:24px auto;padding:0 16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:14px;box-shadow:0 3px 14px #102c4308}}.metric{{font-size:27px;font-weight:750;letter-spacing:-.5px}}.label{{color:var(--muted);font-size:13px}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:15px;margin:12px 0 6px}}.pill{{display:inline-block;border-radius:999px;padding:3px 9px;background:#e5f2f7;color:#135a73;font-size:12px;font-weight:650}}.warning{{background:var(--warn);border-color:#f0d99a}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}}th{{color:var(--muted);font-weight:650}}ul{{padding-left:21px;margin:8px 0}}li{{margin:5px 0}}.muted{{color:var(--muted)}}footer{{color:var(--muted);font-size:12px;padding:12px 0 30px}}@media(max-width:600px){{header h1{{font-size:23px}}.metric{{font-size:23px}}}}
</style></head><body>
<header><h1>ForgeFlex</h1><p>Industrial energy intelligence · sensor validation priorities · tariff-aware scheduling</p></header>
<main>
<div class="card warning"><span class="pill">Evidence boundary</span><p><strong>This report combines different evidence types.</strong> HIPE is a public historical dataset; sensor rankings are retrospective heuristics; scheduling metrics are simulation outputs. They are not proof of performance or savings at a real factory.</p></div>
<section class="grid">
<div class="card"><div class="label">HIPE aligned rows</div><div class="metric">{_esc(hipe.get('rows_after_alignment','—'))}</div><div class="label">Public dataset · {_esc(hipe.get('frequency','frequency not recorded'))}</div></div>
<div class="card"><div class="label">Power series</div><div class="metric">{len(hipe.get('columns',[]) or [])}</div><div class="label">Includes MainTerminal</div></div>
<div class="card"><div class="label">Eligible sensing candidates</div><div class="metric">{_esc(sensor.get('eligible_machines','—'))}</div><div class="label">Retrospective prioritization</div></div>
<div class="card"><div class="label">Simulated cost change</div><div class="metric">{_num(cost_pct,2)}%</div><div class="label">Optimized vs baseline; negative is lower</div></div>
</section>
<section class="card"><h2>1 · Historical power evidence</h2><p><span class="pill">{_esc(hipe.get('data_type','not available'))}</span></p>
<p>Dataset: <strong>{_esc(hipe.get('dataset','not available'))}</strong>. The aligned data uses fixed-width UTC bins and preserves missing bins.</p>
<p class="muted">Missingness by series is recorded in the JSON report; no inference about the user's own factory should be drawn from this public dataset.</p>
<h3>Activity audit</h3><p>Audit status: <strong>{_esc(activity.get('data_type','not available'))}</strong>. Threshold: {_esc(activity.get('epsilon_kw','not recorded'))} kW.</p>
<p class="muted">Activity is a heuristic based on a power threshold, not official machine ON/OFF ground truth.</p></section>
<section class="card"><h2>2 · Aggregate-only inference validation</h2><table><thead><tr><th>Model</th><th>Weighted MAE (kW)</th><th>Active-only weighted MAE (kW)</th></tr></thead><tbody>{wf_rows}</tbody></table>
<p class="muted">Scores summarize saved chronological held-out folds. They do not establish live deployment accuracy or cross-factory generalization.</p></section>
<section class="card"><h2>3 · Candidate submeters to validate</h2><p>Selected under the configured budget: <strong>{_esc(sensor.get('budget_slots','—'))}</strong>. These are validation priorities, not automatic installation instructions.</p><ol>{selected_html}</ol><p class="muted">The score is a relative percentile heuristic over eligible machines. It is not ROI, calibrated uncertainty, physical sensor value, or predicted energy savings.</p></section>
<section class="card"><h2>4 · Tariff-aware production scheduling</h2><p><span class="pill">{_esc(sched.get('data_type','not available'))}</span> Jobs source: {_esc(sched.get('jobs_input_source','not recorded'))}; tariff source: {_esc(sched.get('tariff_input_source','not recorded'))}.</p>
<table><thead><tr><th>Metric</th><th>Baseline</th><th>Tariff-aware</th><th>Optimized − baseline</th></tr></thead><tbody>
<tr><td>Energy cost (illustrative units)</td><td>{_num(sched_base.get('total_energy_cost'),2)}</td><td>{_num(sched_opt.get('total_energy_cost'),2)}</td><td>{_num(cost_delta,2)}</td></tr>
<tr><td>Energy (kWh)</td><td>{_num(sched_base.get('total_energy_kwh'),2)}</td><td>{_num(sched_opt.get('total_energy_kwh'),2)}</td><td>{_num((sched.get('delta_optimized_minus_baseline') or {}).get('energy_kwh'),2)}</td></tr>
<tr><td>Peak scheduled demand (kW)</td><td>{_num(sched_base.get('peak_aggregate_demand_kw'),2)}</td><td>{_num(sched_opt.get('peak_aggregate_demand_kw'),2)}</td><td>{_num(peak_delta,2)}</td></tr>
<tr><td>Jobs scheduled / requested</td><td>{_esc(sched_base.get('jobs_scheduled','—'))} / {_esc(sched_base.get('jobs_requested','—'))}</td><td>{_esc(sched_opt.get('jobs_scheduled','—'))} / {_esc(sched_opt.get('jobs_requested','—'))}</td><td>—</td></tr>
<tr><td>Deadline misses</td><td>{_esc(sched_base.get('deadline_misses_among_scheduled_jobs','—'))}</td><td>{_esc(sched_opt.get('deadline_misses_among_scheduled_jobs','—'))}</td><td>—</td></tr>
</tbody></table>
<p class="muted">The current scheduler does not consume the sensor ranking or inference residuals as constraints. These sections are consolidated for review, not causally coupled. The scheduler is a greedy heuristic, not a global optimizer.</p></section>
<section class="card warning"><h2>Interpretation guardrails</h2><ul>{caveat_html}</ul></section>
<section class="card"><h2>Input artifact status</h2><ul>{warning_html}</ul></section>
<footer>Generated by ForgeFlex from local artifacts. Inspect decision_report.json for full provenance and decision_summary.csv for machine-readable KPIs.</footer>
</main></body></html>'''


def run_decision_report(
    output_dir: str | Path,
    hipe_audit_json: str | Path | None = "outputs/hipe/hipe_audit.json",
    activity_audit_json: str | Path | None = "outputs/hipe_activity_audit/hipe_activity_audit.json",
    walkforward_metrics_csv: str | Path | None = "outputs/hipe_walkforward/hipe_walkforward_metrics.csv",
    sensor_summary_json: str | Path | None = "outputs/sensor_placement_v1/sensor_placement_summary.json",
    sensor_recommendations_csv: str | Path | None = "outputs/sensor_placement_v1/sensor_placement_recommendations.csv",
    scheduling_summary_json: str | Path | None = "outputs/scheduling_demo_v1/scheduling_summary.json",
) -> dict[str, Any]:
    """Combine existing artifacts into JSON, CSV and standalone HTML reports."""
    missing: list[str] = []
    hipe = _read_json(hipe_audit_json, "HIPE audit", missing)
    activity = _read_json(activity_audit_json, "activity audit", missing)
    wf_metrics = _read_csv(walkforward_metrics_csv, "walk-forward metrics", missing)
    sensor = _read_json(sensor_summary_json, "sensor-placement summary", missing)
    recommendations = _read_csv(sensor_recommendations_csv, "sensor recommendations", missing)
    scheduling = _read_json(scheduling_summary_json, "scheduling summary", missing)
    wf = _walkforward_summary(wf_metrics)

    selected: list[str] = []
    if sensor and isinstance(sensor.get("selected_for_submeter_validation"), list):
        selected = [str(x) for x in sensor["selected_for_submeter_validation"]]
    candidate_rows: list[dict[str, Any]] = []
    if recommendations is not None and "machine" in recommendations.columns:
        for _, row in recommendations.iterrows():
            candidate_rows.append({
                "machine": str(row.get("machine", "")),
                "eligible_for_ranking": bool(row.get("eligible_for_ranking", False)) if pd.notna(row.get("eligible_for_ranking", False)) else False,
                "priority_score_0_to_100": float(row["priority_score_0_to_100"]) if "priority_score_0_to_100" in row and pd.notna(row["priority_score_0_to_100"]) else None,
                "recommendation_rank": int(row["recommendation_rank"]) if "recommendation_rank" in row and pd.notna(row["recommendation_rank"]) else None,
                "recommendation_status": str(row.get("recommendation_status", "not recorded")),
                "active_test_samples_across_folds": int(row["active_test_samples_across_folds"]) if "active_test_samples_across_folds" in row and pd.notna(row["active_test_samples_across_folds"]) else None,
            })

    report: dict[str, Any] = {
        "title": "ForgeFlex integrated decision report",
        "report_version": 1,
        "generated_by": "forgeflex.decision_report",
        "evidence_types": {
            "hipe": "REAL_PUBLIC_DATASET_DIAGNOSTIC",
            "activity_audit": "REAL_PUBLIC_DATASET_DIAGNOSTIC",
            "walkforward": "CHRONOLOGICAL_HELD_OUT_MODEL_EVALUATION",
            "sensor_placement": "RETROSPECTIVE_VALIDATION_PRIORITIZATION",
            "scheduling": "SIMULATION_ONLY",
        },
        "hipe": hipe,
        "activity_audit": activity,
        "walkforward": wf,
        "sensor_placement": ({**sensor, "selected_for_submeter_validation": selected,
                              "candidate_rows": candidate_rows} if sensor else None),
        "scheduling": scheduling,
        "missing_or_unreadable_inputs": missing,
        "interpretation_guardrails": [
            "HIPE is a public one-week dataset, not a measurement of the user's factory.",
            "The activity threshold is a heuristic and is not official machine-state ground truth.",
            "Sensor-placement scores rank candidates for submeter validation; they do not estimate ROI, payback, or causal sensor value.",
            "Scheduling outputs are simulations. Synthetic job/tariff results must not be reported as measured factory savings.",
            "A lower simulated tariff bill is not equivalent to lower energy consumption.",
            "The current modules are consolidated in this report but are not yet coupled into one optimization objective.",
        ],
    }

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "decision_report.json"
    csv_path = outdir / "decision_summary.csv"
    html_path = outdir / "decision_report.html"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    sched = scheduling or {}
    base = sched.get("baseline") or {}
    opt = sched.get("optimized") or {}
    delta = sched.get("delta_optimized_minus_baseline") or {}
    pct = sched.get("percent_change_optimized_vs_baseline") or {}
    hipe_obj = hipe or {}
    sensor_obj = sensor or {}
    wf_models = {row["model"]: row for row in (wf or {}).get("by_model", [])}
    summary_rows = [
        {"section": "data", "metric": "hipe_rows_after_alignment", "value": hipe_obj.get("rows_after_alignment"), "unit": "rows", "evidence_type": "REAL_PUBLIC_DATASET_DIAGNOSTIC"},
        {"section": "data", "metric": "hipe_power_series_count", "value": len(hipe_obj.get("columns", []) or []), "unit": "series", "evidence_type": "REAL_PUBLIC_DATASET_DIAGNOSTIC"},
        {"section": "model", "metric": "ridge_weighted_mae_kw", "value": next((r.get("weighted_mae_kw") for r in wf_models.values() if r.get("model") == "ridge"), None), "unit": "kW", "evidence_type": "CHRONOLOGICAL_HELD_OUT_MODEL_EVALUATION"},
        {"section": "sensing", "metric": "eligible_machines", "value": sensor_obj.get("eligible_machines"), "unit": "machines", "evidence_type": "RETROSPECTIVE_VALIDATION_PRIORITIZATION"},
        {"section": "sensing", "metric": "selected_candidates", "value": ";".join(selected), "unit": "machine names", "evidence_type": "RETROSPECTIVE_VALIDATION_PRIORITIZATION"},
        {"section": "schedule", "metric": "baseline_cost", "value": base.get("total_energy_cost"), "unit": "illustrative units", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "optimized_cost", "value": opt.get("total_energy_cost"), "unit": "illustrative units", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "optimized_cost_change_percent", "value": pct.get("energy_cost"), "unit": "%", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "energy_change_kwh", "value": delta.get("energy_kwh"), "unit": "kWh", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "peak_demand_change_kw", "value": delta.get("peak_aggregate_demand_kw"), "unit": "kW", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "baseline_deadline_misses", "value": base.get("deadline_misses_among_scheduled_jobs"), "unit": "jobs", "evidence_type": "SIMULATION_ONLY"},
        {"section": "schedule", "metric": "optimized_deadline_misses", "value": opt.get("deadline_misses_among_scheduled_jobs"), "unit": "jobs", "evidence_type": "SIMULATION_ONLY"},
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    html_path.write_text(_html_report(report), encoding="utf-8")
    return {"json_report": str(json_path), "csv_summary": str(csv_path), "html_report": str(html_path),
            "missing_or_unreadable_inputs": missing, "evidence_types": report["evidence_types"]}
