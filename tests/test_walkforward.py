import json
import numpy as np
import pandas as pd
import pytest

from forgeflex.walkforward import run_walkforward


def make_csv(path, n=1000):
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


def test_walkforward_writes_metrics_and_chronological_folds(tmp_path):
    input_path = tmp_path / "aligned.csv"
    make_csv(input_path)
    result = run_walkforward(input_path, tmp_path / "out")
    summary = json.loads((tmp_path / "out" / "hipe_walkforward_summary.json").read_text())
    metrics = pd.read_csv(tmp_path / "out" / "hipe_walkforward_metrics.csv")
    assert result["summary"]["folds_evaluated"] == 7
    assert summary["method"].startswith("Expanding-window chronological")
    assert set(metrics["model"].dropna()) == {"ridge", "zero_baseline", "train_mean_baseline"}
    ridge = metrics[metrics["model"] == "ridge"]
    assert (ridge["test_start_index"] >= ridge["train_end_index_exclusive"]).all()
    assert ridge["active_fraction_test"].notna().all()
    assert "active_only_mae_kw" in metrics.columns


def test_walkforward_rejects_bad_alpha(tmp_path):
    input_path = tmp_path / "aligned.csv"
    make_csv(input_path)
    with pytest.raises(ValueError, match="alpha"):
        run_walkforward(input_path, tmp_path / "out", alpha=0)


def test_walkforward_requires_expected_columns(tmp_path):
    input_path = tmp_path / "bad.csv"
    pd.DataFrame({"time": [1, 2], "power": [3, 4]}).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="timestamp_utc"):
        run_walkforward(input_path, tmp_path / "out")
