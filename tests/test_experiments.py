import json
import pandas as pd
import pytest

from forgeflex.experiments import run_scheduling_experiments


def test_experiment_suite_runs_paired_scenarios_and_writes_artifacts(tmp_path):
    result = run_scheduling_experiments(
        output_dir=tmp_path / "experiments", seeds=[3], job_counts=[4],
        horizon_slots=12, slot_hours=1.0,
    )
    assert result["summary"]["data_type"] == "SIMULATION_ONLY"
    assert result["summary"]["scenario_count"] == 2
    detail = pd.read_csv(result["results_csv"])
    assert set(detail["tariff_condition"]) == {"illustrative_tou", "illustrative_flat"}
    assert (detail["data_type"] == "SIMULATION_ONLY").all()
    assert (detail["baseline_production_units"] == detail["optimized_production_units"]).all()
    saved = json.loads(open(result["summary_json"], encoding="utf-8").read())
    assert saved["scenario_count"] == 2


def test_experiment_suite_rejects_empty_seeds(tmp_path):
    with pytest.raises(ValueError, match="seeds and job_counts"):
        run_scheduling_experiments(tmp_path / "out", seeds=[], job_counts=[2], horizon_slots=12)


def test_experiment_suite_rejects_too_short_horizon(tmp_path):
    with pytest.raises(ValueError, match="horizon_slots"):
        run_scheduling_experiments(tmp_path / "out", seeds=[1], job_counts=[2], horizon_slots=4)
