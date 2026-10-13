"""Temporal activity audit for raw and aligned HIPE power series.

This diagnostic does not modify source data, fill missing values, infer machine
states, or claim energy savings. Activity threshold is a configurable heuristic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED = {"SensorDateTime", "Machine", "P_kW"}


def _stats(values: pd.Series, epsilon_kw: float) -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "n_valid": 0, "min_kw": None, "max_kw": None, "mean_kw": None,
            "std_kw": None, "nonzero_fraction": None,
            "active_fraction_above_epsilon": None,
        }
    return {
        "n_valid": int(len(x)),
        "min_kw": float(x.min()),
        "max_kw": float(x.max()),
        "mean_kw": float(x.mean()),
        "std_kw": float(x.std(ddof=0)),
        "nonzero_fraction": float((x != 0).mean()),
        "active_fraction_above_epsilon": float((x.abs() > epsilon_kw).mean()),
    }


def _plot_daily(summary_csv: Path, output_png: Path, epsilon_kw: float) -> str | None:
    """Write a daily activity heatmap if matplotlib is installed; otherwise skip."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    df = pd.read_csv(summary_csv)
    if df.empty or "active_fraction_above_epsilon" not in df.columns:
        return None
    pivot = df.pivot(index="machine", columns="utc_date", values="active_fraction_above_epsilon")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(pivot.columns)), max(4, 0.45 * len(pivot.index))))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
    ax.set_title(f"Daily fraction |P_kW| > {epsilon_kw:g} kW (heuristic)")
    ax.set_xlabel("UTC date")
    ax.set_ylabel("Machine / series")
    fig.colorbar(im, ax=ax, label="Active fraction")
    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return str(output_png)


def run_activity_audit(
    data_dir: str | Path,
    aligned_csv: str | Path,
    output_dir: str | Path,
    epsilon_kw: float = 0.05,
) -> dict[str, Any]:
    """Summarize raw per-machine and aligned per-day activity in UTC."""
    data_dir, aligned_csv, output_dir = Path(data_dir), Path(aligned_csv), Path(output_dir)
    if epsilon_kw < 0:
        raise ValueError("epsilon_kw must be >= 0")
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw HIPE directory not found: {data_dir}")
    if not aligned_csv.exists():
        raise FileNotFoundError(f"Aligned CSV not found: {aligned_csv}")
    files = sorted(data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found under {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_daily_rows: list[dict[str, Any]] = []
    raw_file_rows: list[dict[str, Any]] = []
    for path in files:
        try:
            df = pd.read_csv(path, usecols=list(REQUIRED))
        except ValueError as exc:
            raise ValueError(f"{path.name} must contain columns {sorted(REQUIRED)}") from exc
        df["SensorDateTime"] = pd.to_datetime(df["SensorDateTime"], utc=True, errors="coerce")
        df["P_kW"] = pd.to_numeric(df["P_kW"], errors="coerce")
        invalid_time = int(df["SensorDateTime"].isna().sum())
        df = df.dropna(subset=["SensorDateTime", "Machine", "P_kW"]).copy()
        if df.empty:
            raw_file_rows.append({"file": path.name, "status": "no_valid_rows", "invalid_timestamp_rows": invalid_time})
            continue
        machines = sorted(df["Machine"].astype(str).unique())
        machine_label = machines[0] if len(machines) == 1 else ";".join(machines)
        raw_file_rows.append({
            "file": path.name, "machine": machine_label, "status": "ok",
            "rows_in_file": int(len(pd.read_csv(path, usecols=["SensorDateTime"]))),
            "valid_rows": int(len(df)), "invalid_timestamp_rows": invalid_time,
            "first_timestamp_utc": df["SensorDateTime"].min().isoformat(),
            "last_timestamp_utc": df["SensorDateTime"].max().isoformat(),
            "min_kw": float(df["P_kW"].min()), "max_kw": float(df["P_kW"].max()),
        })
        df["utc_date"] = df["SensorDateTime"].dt.strftime("%Y-%m-%d")
        for day, group in df.groupby("utc_date", sort=True):
            row = {"source": "raw", "machine": machine_label, "utc_date": day, **_stats(group["P_kW"], epsilon_kw)}
            raw_daily_rows.append(row)

    raw_daily = pd.DataFrame(raw_daily_rows)
    raw_daily_path = output_dir / "hipe_raw_daily_activity.csv"
    raw_files_path = output_dir / "hipe_raw_file_audit.csv"
    raw_daily.to_csv(raw_daily_path, index=False)
    pd.DataFrame(raw_file_rows).to_csv(raw_files_path, index=False)

    aligned = pd.read_csv(aligned_csv)
    if "timestamp_utc" not in aligned.columns:
        raise ValueError(f"{aligned_csv} must contain timestamp_utc column")
    aligned["timestamp_utc"] = pd.to_datetime(aligned["timestamp_utc"], utc=True, errors="coerce")
    invalid_aligned = int(aligned["timestamp_utc"].isna().sum())
    aligned = aligned.dropna(subset=["timestamp_utc"]).copy()
    aligned["utc_date"] = aligned["timestamp_utc"].dt.strftime("%Y-%m-%d")
    aligned_daily_rows: list[dict[str, Any]] = []
    series_cols = [c for c in aligned.columns if c not in {"timestamp_utc", "utc_date"}]
    for machine in series_cols:
        for day, group in aligned.groupby("utc_date", sort=True):
            stats = _stats(group[machine], epsilon_kw)
            aligned_daily_rows.append({"source": "aligned", "machine": machine, "utc_date": day, **stats,
                                       "n_bins": int(len(group)), "missing_fraction": float(group[machine].isna().mean())})
    aligned_daily = pd.DataFrame(aligned_daily_rows)
    aligned_daily_path = output_dir / "hipe_aligned_daily_activity.csv"
    aligned_daily.to_csv(aligned_daily_path, index=False)

    raw_plot = _plot_daily(raw_daily_path, output_dir / "hipe_raw_activity_heatmap.png", epsilon_kw)
    aligned_plot = _plot_daily(aligned_daily_path, output_dir / "hipe_aligned_activity_heatmap.png", epsilon_kw)

    # Identify machine-days with a discrepancy in raw vs aligned active fraction.
    merged = raw_daily.merge(
        aligned_daily, on=["machine", "utc_date"], how="outer", suffixes=("_raw", "_aligned")
    )
    if not merged.empty:
        merged["active_fraction_difference_aligned_minus_raw"] = (
            merged["active_fraction_above_epsilon_aligned"] - merged["active_fraction_above_epsilon_raw"]
        )
    discrepancy_path = output_dir / "hipe_raw_vs_aligned_daily.csv"
    merged.to_csv(discrepancy_path, index=False)

    summary = {
        "data_type": "REAL_PUBLIC_DATASET_DIAGNOSTIC",
        "dataset": "HIPE one-week archive",
        "epsilon_kw": epsilon_kw,
        "epsilon_note": "Heuristic threshold using absolute power; not an official machine-state label.",
        "raw_files_found": len(files),
        "raw_files_with_valid_rows": sum(1 for row in raw_file_rows if row.get("status") == "ok"),
        "raw_file_audit_csv": str(raw_files_path),
        "raw_daily_activity_csv": str(raw_daily_path),
        "aligned_daily_activity_csv": str(aligned_daily_path),
        "raw_vs_aligned_daily_csv": str(discrepancy_path),
        "raw_heatmap_png": raw_plot,
        "aligned_heatmap_png": aligned_plot,
        "aligned_rows_after_invalid_timestamps_removed": int(len(aligned)),
        "aligned_invalid_timestamp_rows": invalid_aligned,
        "aligned_series": series_cols,
        "warnings": [
            "Audit only; no source data was modified.",
            "Daily grouping uses UTC dates to match the current alignment pipeline.",
            "Raw and aligned activity fractions may differ because binning, missing bins, and timestamps differ.",
            "Activity threshold is heuristic and must not be presented as official machine ON/OFF ground truth.",
            "A temporal split must not be selected only to improve model scores.",
        ],
    }
    summary_path = output_dir / "hipe_activity_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_json": str(summary_path), "summary": summary}
