import json
from pathlib import Path

import pandas as pd

from forgeflex.dashboard_utils import metric_value, numeric_columns, read_csv, read_json


def test_read_json_missing_and_valid(tmp_path: Path):
    assert read_json(tmp_path / "missing.json") is None
    path = tmp_path / "valid.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert read_json(path) == {"a": 1}


def test_read_csv_and_numeric_columns(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("timestamp,power_kw,machine\n2026-01-01,1.2,A\n", encoding="utf-8")
    frame = read_csv(path)
    assert numeric_columns(frame, {"timestamp"}) == ["power_kw"]
    assert read_csv(tmp_path / "missing.csv").empty


def test_metric_value_handles_missing_and_bad_values():
    assert metric_value({"baseline": {"cost": 12}}, "baseline", "cost") == 12.0
    assert metric_value({}, "baseline", "cost") is None
    assert metric_value({"baseline": {"cost": "n/a"}}, "baseline", "cost") is None
