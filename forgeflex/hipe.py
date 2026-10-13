"""HIPE data inspection and conservative time alignment utilities.

This module reads only SensorDateTime, Machine and P_kW. It does not infer
missing readings, clip negative values, or claim energy savings.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = {"SensorDateTime", "Machine", "P_kW"}


def inspect_hipe_csv(path: Path, chunksize: int = 250_000) -> dict[str, Any]:
    """Inspect one HIPE CSV in chunks without loading its wide sensor schema."""
    path = Path(path)
    rows = 0
    valid_timestamps = 0
    invalid_timestamps = 0
    machines: set[str] = set()
    p_min = None
    p_max = None
    negative_count = 0
    first_timestamp = None
    last_timestamp = None
    all_times: list[pd.Series] = []

    try:
        iterator = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS), chunksize=chunksize)
        for chunk in iterator:
            rows += len(chunk)
            machines.update(str(v) for v in chunk["Machine"].dropna().unique())
            power = pd.to_numeric(chunk["P_kW"], errors="coerce")
            if power.notna().any():
                chunk_min, chunk_max = float(power.min()), float(power.max())
                p_min = chunk_min if p_min is None else min(p_min, chunk_min)
                p_max = chunk_max if p_max is None else max(p_max, chunk_max)
                negative_count += int((power < 0).sum())

            timestamps = pd.to_datetime(chunk["SensorDateTime"], utc=True, errors="coerce")
            valid = timestamps.dropna()
            valid_timestamps += len(valid)
            invalid_timestamps += int(timestamps.isna().sum())
            if len(valid):
                chunk_first, chunk_last = valid.min(), valid.max()
                first_timestamp = chunk_first if first_timestamp is None else min(first_timestamp, chunk_first)
                last_timestamp = chunk_last if last_timestamp is None else max(last_timestamp, chunk_last)
                # Keep timestamps only for interval estimation; HIPE one-week files fit comfortably.
                all_times.append(valid.reset_index(drop=True))
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: expected columns {sorted(REQUIRED_COLUMNS)}; "
            f"check the CSV header. Original error: {exc}"
        ) from exc

    if rows == 0:
        raise ValueError(f"{path} contains no data rows.")

    times = pd.concat(all_times, ignore_index=True).sort_values() if all_times else pd.Series(dtype="datetime64[ns, UTC]")
    diffs = times.diff().dt.total_seconds()
    positive_diffs = diffs[diffs > 0]
    return {
        "file": path.name,
        "rows": rows,
        "machine_values": sorted(machines),
        "valid_timestamps": valid_timestamps,
        "invalid_timestamps": invalid_timestamps,
        "first_timestamp_utc": first_timestamp.isoformat() if first_timestamp is not None else None,
        "last_timestamp_utc": last_timestamp.isoformat() if last_timestamp is not None else None,
        "median_positive_interval_seconds": round(float(positive_diffs.median()), 6) if len(positive_diffs) else None,
        "p_kw_min": p_min,
        "p_kw_max": p_max,
        "negative_power_rows": negative_count,
    }


def load_hipe_power(
    data_dir: str | Path,
    output_dir: str | Path,
    frequency: str = "10s",
) -> dict[str, Any]:
    """Create aligned main-terminal and machine power series from HIPE CSV files.

    Power values are averaged within fixed UTC bins. No interpolation or
    forward-fill is performed. The output is an analysis-ready dataset,
    not a claim that each sensor was measured at exactly the same instant.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    power_series: dict[str, pd.Series] = {}
    audits: list[dict[str, Any]] = []
    for path in files:
        # Audit first; also verifies required columns.
        audit = inspect_hipe_csv(path)
        audits.append(audit)

        df = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
        df["SensorDateTime"] = pd.to_datetime(
            df["SensorDateTime"], utc=True, errors="coerce"
        )
        df["P_kW"] = pd.to_numeric(df["P_kW"], errors="coerce")
        df = df.dropna(subset=["SensorDateTime", "Machine", "P_kW"])
        if df.empty:
            raise ValueError(f"{path.name} has no rows with valid timestamps, machine, and P_kW.")

        unique_machines = df["Machine"].astype(str).unique()
        if len(unique_machines) != 1:
            raise ValueError(
                f"{path.name} contains {len(unique_machines)} Machine values: "
                f"{unique_machines.tolist()}. Expected one machine per file."
            )
        machine = str(unique_machines[0])
        if machine in power_series:
            raise ValueError(f"Duplicate machine series found for {machine!r}.")
        df = df.sort_values("SensorDateTime")
        # Duplicate timestamps are allowed; resampling averages them within the bin.
        sampled = (
            df.set_index("SensorDateTime")["P_kW"]
            .resample(frequency, origin="epoch")
            .mean()
            .rename(machine)
        )
        power_series[machine] = sampled

    if "MainTerminal" not in power_series:
        raise ValueError(
            "Could not find Machine='MainTerminal'. Verify the main-terminal CSV "
            "and its Machine column before proceeding."
        )

    aligned = pd.concat(power_series.values(), axis=1).sort_index()
    aligned.index.name = "timestamp_utc"
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / f"hipe_power_{frequency.replace('/', '_')}.csv"
    aligned.reset_index().to_csv(aligned_path, index=False)

    # Summarize missingness without filling any missing bins.
    missingness = {
        str(col): round(float(aligned[col].isna().mean()), 6)
        for col in aligned.columns
    }
    audit_path = output_dir / "hipe_audit.json"
    summary = {
        "data_type": "REAL_PUBLIC_DATASET",
        "dataset": "HIPE one-week archive",
        "source_directory": str(data_dir),
        "frequency": frequency,
        "alignment": "UTC fixed-width bins; mean P_kW per bin; missing bins remain missing",
        "rows_after_alignment": int(len(aligned)),
        "columns": list(aligned.columns),
        "missing_fraction_by_series": missingness,
        "files": audits,
        "warnings": [
            "This is a public industrial dataset, not a measurement of the user's own factory.",
            "No missing values were interpolated or forward-filled.",
            "Negative P_kW values were preserved for audit; inspect sensor offsets before any clipping.",
            "Aligned bins average asynchronous measurements and should not be described as exact simultaneous readings.",
            "No savings, energy-reduction, or scheduling improvement is calculated by this command."
        ],
        "aligned_csv": str(aligned_path),
    }
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"audit_json": str(audit_path), "aligned_csv": str(aligned_path), "summary": summary}
