import numpy as np
import pandas as pd
import pytest

from forgeflex.inference import run_inference, _features, _metrics


def test_features_use_aggregate_and_timestamp_only():
    df = pd.DataFrame({
        "timestamp_utc": pd.date_range("2020-01-01", periods=40, freq="10s", tz="UTC").astype(str),
        "MainTerminal": np.linspace(10, 20, 40),
        "MachineA": np.linspace(2, 5, 40),
    })
    feats = _features(df)
    assert "aggregate_now_kw" in feats.columns
    assert "MachineA" not in feats.columns
    assert feats.iloc[20:].notna().all().all()


def test_metrics_known_values_and_undefined_denominator():
    metrics = _metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics["mae_kw"] == pytest.approx(0.5)
    assert metrics["rmse_kw"] == pytest.approx(np.sqrt(0.5))
    zero = _metrics(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert zero["r2"] is None
    assert zero["normalized_disaggregation_error"] is None
    assert zero["signed_energy_relative_error"] is None


def test_inference_chronological_test_predictions_and_baselines(tmp_path):
    n = 240
    ts = pd.date_range("2020-01-01", periods=n, freq="10s", tz="UTC")
    agg = 10 + 2 * np.sin(np.arange(n) / 10)
    target = 0.3 * agg + 0.5 * np.sin(np.arange(n) / 8)
    df = pd.DataFrame({
        "timestamp_utc": ts.astype(str),
        "MainTerminal": agg,
        "MachineA": target,
    })
    input_path = tmp_path / "aligned.csv"
    df.to_csv(input_path, index=False)
    result = run_inference(input_path, tmp_path / "out", train_fraction=0.7)
    pred = pd.read_csv(result["predictions_csv"])
    assert result["summary"]["data_type"] == "REAL_PUBLIC_DATASET_EVALUATION"
    assert result["summary"]["split_index"] == int(n * 0.7)
    assert pred["MachineA__ridge_kw"].iloc[:int(n * 0.7)].isna().all()
    assert pred["MachineA__ridge_kw"].iloc[int(n * 0.7):].notna().sum() > 20
    machine = result["summary"]["metrics_by_machine"]["MachineA"]
    assert machine["status"] == "evaluated"
    assert "zero_baseline" in machine["metrics"]
    assert "train_mean_baseline" in machine["metrics"]
    assert "active_fraction_above_epsilon" in machine["target_diagnostics"]["test"]


def test_inference_rejects_missing_terminal(tmp_path):
    input_path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp_utc": ["2020-01-01"], "MachineA": [1]}).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="MainTerminal"):
        run_inference(input_path, tmp_path / "out")
