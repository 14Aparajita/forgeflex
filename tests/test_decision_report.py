import json
from pathlib import Path

import pandas as pd

from forgeflex.decision_report import run_decision_report


def test_decision_report_builds_json_csv_html_and_preserves_evidence_types(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "hipe.json").write_text(json.dumps({
        "data_type": "REAL_PUBLIC_DATASET", "dataset": "HIPE one-week archive",
        "frequency": "10s", "rows_after_alignment": 100, "columns": ["MainTerminal", "M1"],
    }))
    (inputs / "activity.json").write_text(json.dumps({"data_type": "REAL_PUBLIC_DATASET_DIAGNOSTIC", "epsilon_kw": 0.05}))
    pd.DataFrame([
        {"model": "ridge", "status": "evaluated", "mae_kw": 0.5, "test_rows": 100, "n_active_test": 50, "active_only_mae_kw": 0.7},
        {"model": "zero_baseline", "status": "evaluated", "mae_kw": 0.8, "test_rows": 100, "n_active_test": 50, "active_only_mae_kw": 1.0},
    ]).to_csv(inputs / "metrics.csv", index=False)
    (inputs / "sensor.json").write_text(json.dumps({
        "data_type": "RETROSPECTIVE_VALIDATION_PRIORITIZATION", "eligible_machines": 1,
        "budget_slots": 1, "selected_for_submeter_validation": ["M1"],
    }))
    pd.DataFrame([{"machine": "M1", "eligible_for_ranking": True, "priority_score_0_to_100": 75,
                   "recommendation_rank": 1, "recommendation_status": "recommend_validate_submeter_candidate",
                   "active_test_samples_across_folds": 120}]).to_csv(inputs / "recommendations.csv", index=False)
    (inputs / "schedule.json").write_text(json.dumps({
        "data_type": "SIMULATION_ONLY", "jobs_input_source": "SYNTHETIC_DEMO_INPUT",
        "tariff_input_source": "SYNTHETIC_ILLUSTRATIVE_TARIFF",
        "baseline": {"total_energy_cost": 100, "total_energy_kwh": 10, "peak_aggregate_demand_kw": 5,
                     "jobs_scheduled": 2, "jobs_requested": 2, "deadline_misses_among_scheduled_jobs": 0},
        "optimized": {"total_energy_cost": 90, "total_energy_kwh": 10, "peak_aggregate_demand_kw": 4,
                      "jobs_scheduled": 2, "jobs_requested": 2, "deadline_misses_among_scheduled_jobs": 0},
        "delta_optimized_minus_baseline": {"energy_cost": -10, "energy_kwh": 0, "peak_aggregate_demand_kw": -1},
        "percent_change_optimized_vs_baseline": {"energy_cost": -10},
    }))
    result = run_decision_report(
        tmp_path / "out", inputs / "hipe.json", inputs / "activity.json", inputs / "metrics.csv",
        inputs / "sensor.json", inputs / "recommendations.csv", inputs / "schedule.json",
    )
    assert not result["missing_or_unreadable_inputs"]
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert report["scheduling"]["data_type"] == "SIMULATION_ONLY"
    assert report["sensor_placement"]["selected_for_submeter_validation"] == ["M1"]
    assert report["walkforward"]["by_model"][0]["weighted_mae_kw"] is not None
    assert Path(result["csv_summary"]).exists()
    html = Path(result["html_report"]).read_text(encoding="utf-8")
    assert "Evidence boundary" in html
    assert "not proof of performance" in html


def test_decision_report_missing_inputs_is_nonfatal_and_explicit(tmp_path):
    result = run_decision_report(
        tmp_path / "out", tmp_path / "no-hipe.json", tmp_path / "no-activity.json",
        tmp_path / "no-metrics.csv", tmp_path / "no-sensor.json", tmp_path / "no-rec.csv",
        tmp_path / "no-schedule.json",
    )
    assert len(result["missing_or_unreadable_inputs"]) == 6
    report = json.loads(Path(result["json_report"]).read_text(encoding="utf-8"))
    assert report["hipe"] is None
    assert Path(result["html_report"]).exists()
