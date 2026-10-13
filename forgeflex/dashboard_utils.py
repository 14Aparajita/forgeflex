"""Dependency-light helpers used by the ForgeFlex interactive demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: str | Path) -> dict[str, Any] | None:
    """Read a JSON object; return None for missing, unreadable, or non-object files."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, returning an empty frame on file/parse errors."""
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return pd.DataFrame()


def numeric_columns(frame: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Return numeric columns, optionally excluding named identifiers/timestamps."""
    excluded = {str(x).lower() for x in (exclude or set())}
    return [
        str(column)
        for column in frame.columns
        if str(column).lower() not in excluded
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def metric_value(summary: dict[str, Any] | None, section: str, key: str) -> float | None:
    """Safely extract a numeric scheduling metric from a nested summary."""
    if not isinstance(summary, dict):
        return None
    part = summary.get(section)
    if not isinstance(part, dict):
        return None
    value = part.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None
