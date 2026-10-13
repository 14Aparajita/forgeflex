import json
from pathlib import Path

import pandas as pd

from forgeflex.activity_audit import run_activity_audit


def _write(path: Path, machine: str, times, power):
    pd.DataFrame({"SensorDateTime": times, "Machine": machine, "P_kW": power}).to_csv(path, index=False)


def test_activity_audit_writes_raw_and_aligned_daily_summaries(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "MainTerminal.csv", "MainTerminal",
           ["2017-10-23T00:00:00Z", "2017-10-23T00:00:10Z", "2017-10-24T00:00:00Z"],
           [10.0, 11.0, 0.0])
    _write(raw / "MachineA.csv", "MachineA",
           ["2017-10-23T00:00:00Z", "2017-10-23T00:00:10Z", "2017-10-24T00:00:00Z"],
           [0.0, 0.2, 0.0])
    aligned = tmp_path / "aligned.csv"
    pd.DataFrame({
        "timestamp_utc": ["2017-10-23T00:00:00Z", "2017-10-23T00:00:10Z", "2017-10-24T00:00:00Z"],
        "MainTerminal": [10.0, 11.0, 0.0], "MachineA": [0.0, 0.2, 0.0],
    }).to_csv(aligned, index=False)
    result = run_activity_audit(raw, aligned, tmp_path / "out", 0.05)
    assert Path(result["summary_json"]).exists()
    raw_daily = pd.read_csv(result["summary"]["raw_daily_activity_csv"])
    active = raw_daily[(raw_daily.machine == "MachineA") & (raw_daily.utc_date == "2017-10-23")].iloc[0]
    assert active["active_fraction_above_epsilon"] == 0.5
    aligned_daily = pd.read_csv(result["summary"]["aligned_daily_activity_csv"])
    assert len(aligned_daily) == 4
    saved = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert saved["data_type"] == "REAL_PUBLIC_DATASET_DIAGNOSTIC"


def test_activity_audit_rejects_negative_threshold(tmp_path):
    try:
        run_activity_audit(tmp_path, tmp_path / "missing.csv", tmp_path / "out", -0.1)
    except ValueError as exc:
        assert "epsilon_kw" in str(exc)
    else:
        raise AssertionError("negative threshold should fail")
