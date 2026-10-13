import json
from pathlib import Path

import pandas as pd
import pytest

from forgeflex.hipe import inspect_hipe_csv, load_hipe_power


def write_csv(path: Path, machine: str, times, powers):
    pd.DataFrame({
        "SensorDateTime": times,
        "Machine": [machine] * len(times),
        "P_kW": powers,
    }).to_csv(path, index=False)


def test_hipe_audit_reads_required_columns(tmp_path):
    path = tmp_path / "MachineA.csv"
    write_csv(path, "MachineA",
              ["2017-10-23T00:00:00+02:00", "2017-10-23T00:00:05+02:00"],
              [0.0, 1.0])
    result = inspect_hipe_csv(path)
    assert result["rows"] == 2
    assert result["machine_values"] == ["MachineA"]
    assert result["p_kw_max"] == 1.0
    assert result["invalid_timestamps"] == 0


def test_loader_aligns_main_and_machine_without_filling(tmp_path):
    data = tmp_path / "data"
    out = tmp_path / "out"
    data.mkdir()
    write_csv(data / "MainTerminal.csv", "MainTerminal",
              ["2017-10-23T00:00:00+02:00", "2017-10-23T00:00:10+02:00"],
              [10.0, 12.0])
    write_csv(data / "MachineA.csv", "MachineA",
              ["2017-10-23T00:00:00+02:00"],
              [3.0])
    result = load_hipe_power(data, out, "10s")
    aligned = pd.read_csv(result["aligned_csv"])
    assert "MainTerminal" in aligned.columns
    assert "MachineA" in aligned.columns
    assert result["summary"]["data_type"] == "REAL_PUBLIC_DATASET"
    assert Path(result["audit_json"]).exists()
    assert "No savings" in " ".join(result["summary"]["warnings"])


def test_loader_requires_main_terminal(tmp_path):
    write_csv(tmp_path / "MachineA.csv", "MachineA",
              ["2017-10-23T00:00:00+02:00"], [1.0])
    with pytest.raises(ValueError, match="MainTerminal"):
        load_hipe_power(tmp_path, tmp_path / "out", "10s")
