import json
import numpy as np
import pandas as pd
import pytest

from forgeflex.two_stage import run_two_stage_walkforward, _classification_metrics


def make_csv(path, n=1200):
    ts = pd.date_range("2017-10-23", periods=n, freq="10s", tz="UTC")
    t = np.arange(n)
    terminal = 8 + 2 * np.sin(t / 15) + (t % 70 > 35) * 3
    machine_a = np.where(t % 80 < 25, 0.4 + 0.1 * np.sin(t / 3), 0.0)
    machine_b = np.where(t < n // 2, 0.0, 0.2)
    pd.DataFrame({
        "timestamp_utc": ts.astype(str),
        "MainTerminal": terminal,
        "MachineA": machine_a,
        "MachineB": machine_b,
    }).to_csv(path, index=False)


def test_two_stage_writes_metrics_for_same_chronological_folds(tmp_path):
    input_path = tmp_path / "aligned.csv"
    make_csv(input_path)
    result = run_two_stage_walkforward(input_path, tmp_path / "out")
    summary = json.loads((tmp_path / "out" / "hipe_two_stage_walkforward_summary.json").read_text())
    metrics = pd.read_csv(tmp_path / "out" / "hipe_two_stage_walkforward_metrics.csv")
    assert result["summary"]["folds_evaluated"] == 7
    assert summary["models"] == ["ridge", "two_stage", "zero_baseline", "train_mean_baseline"]
    assert set(metrics["model"].dropna()) == set(summary["models"])
    assert (metrics["test_start_index"] >= metrics["train_end_index_exclusive"]).all()
    two_stage = metrics[metrics["model"] == "two_stage"]
    assert two_stage["activity_precision"].notna().any()
    assert "active_only_mae_kw" in metrics.columns


def test_two_stage_rejects_invalid_activity_threshold(tmp_path):
    input_path = tmp_path / "aligned.csv"
    make_csv(input_path)
    with pytest.raises(ValueError, match="activity_threshold"):
        run_two_stage_walkforward(input_path, tmp_path / "out", activity_threshold=1.0)


def test_classification_metrics_counts():
    result = _classification_metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert result["activity_true_positive"] == 1
    assert result["activity_false_positive"] == 1
    assert result["activity_false_negative"] == 1
    assert result["activity_true_negative"] == 1
    assert result["activity_precision"] == 0.5
    assert result["activity_recall"] == 0.5
    assert result["activity_f1"] == 0.5
