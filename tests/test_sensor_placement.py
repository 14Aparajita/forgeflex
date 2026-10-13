import json

import pandas as pd
import pytest

from forgeflex.sensor_placement import run_sensor_placement


def _write_metrics(path):
    rows = []
    values = {
        # machine: (active-only errors by fold, activity counts, ridge MAEs, zero MAEs, mean MAEs)
        "MachineA": ([1.0, 1.2, 1.1], [80, 90, 100], [1.2, 1.3, 1.25], [0.8, 0.9, 0.85], [1.0, 1.0, 1.0]),
        "MachineB": ([0.2, 0.25, 0.22], [200, 220, 210], [0.3, 0.32, 0.31], [0.6, 0.6, 0.6], [0.5, 0.5, 0.5]),
        "MachineC": ([0.5, None, None], [10, 0, 0], [0.6, 0.1, 0.1], [0.5, 0.1, 0.1], [0.4, 0.1, 0.1]),
    }
    for machine, (active_maes, active_counts, ridge_maes, zero_maes, mean_maes) in values.items():
        for fold in range(1, 4):
            active_n = active_counts[fold - 1]
            test_n = 1000
            frac = active_n / test_n
            common = {
                "fold": fold,
                "machine": machine,
                "status": "evaluated",
                "test_start_index": (fold - 1) * 1000,
                "test_end_index_exclusive": fold * 1000,
                "test_rows": test_n,
                "n_active_test": active_n,
                "active_fraction_test": frac,
                "active_only_mae_kw": active_maes[fold - 1],
            }
            for model, mae in (
                ("ridge", ridge_maes[fold - 1]),
                ("zero_baseline", zero_maes[fold - 1]),
                ("train_mean_baseline", mean_maes[fold - 1]),
            ):
                row = dict(common)
                row["model"] = model
                row["mae_kw"] = mae
                rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_sensor_placement_writes_ranked_recommendations_and_summary(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)

    result = run_sensor_placement(metrics_path, tmp_path / "out", budget=1)
    recommendations = pd.read_csv(result["recommendations_csv"])
    summary = json.loads((tmp_path / "out" / "sensor_placement_summary.json").read_text())

    assert summary["data_type"] == "RETROSPECTIVE_VALIDATION_PRIORITIZATION"
    assert summary["machines_seen"] == 3
    assert summary["eligible_machines"] == 2
    assert len(summary["selected_for_submeter_validation"]) == 1
    assert "priority_score_0_to_100" in recommendations.columns
    assert "active_error_percentile" in recommendations.columns
    assert "activity_percentile" in recommendations.columns
    assert "baseline_disadvantage_percentile" in recommendations.columns
    assert recommendations["recommendation_rank"].notna().sum() == 2
    assert (recommendations["recommendation_status"] == "recommend_validate_submeter_candidate").sum() == 1
    assert "No energy savings" in " ".join(summary["interpretation_notes"])


def test_sensor_placement_excludes_machine_with_insufficient_active_evidence(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)
    result = run_sensor_placement(metrics_path, tmp_path / "out", min_active_samples=100, min_active_folds=2)
    recommendations = pd.read_csv(result["recommendations_csv"]).set_index("machine")

    assert not bool(recommendations.loc["MachineC", "eligible_for_ranking"])
    assert recommendations.loc["MachineC", "priority_tier"] == "insufficient_evidence"
    assert recommendations.loc["MachineC", "recommendation_status"] == "insufficient_validation_evidence"
    assert recommendations.loc["MachineC", "recommendation_rank"] != recommendations.loc["MachineC", "recommendation_rank"]


def test_sensor_placement_rejects_negative_budget(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    _write_metrics(metrics_path)
    with pytest.raises(ValueError, match="budget"):
        run_sensor_placement(metrics_path, tmp_path / "out", budget=-1)


def test_sensor_placement_rejects_missing_required_columns(tmp_path):
    metrics_path = tmp_path / "bad.csv"
    pd.DataFrame({"machine": ["A"], "model": ["ridge"]}).to_csv(metrics_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        run_sensor_placement(metrics_path, tmp_path / "out")


def test_sensor_placement_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="metrics CSV not found"):
        run_sensor_placement(tmp_path / "missing.csv", tmp_path / "out")
