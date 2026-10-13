"""Expanding-window evaluation for aggregate-only machine-power inference.

All test windows occur strictly after their corresponding training windows.
Machine targets are used as labels/metrics, never as model input features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import _features, _fit_ridge_predict, _metrics


def _metric_row(y: np.ndarray, pred: np.ndarray, epsilon_kw: float) -> dict[str, Any]:
    out = _metrics(y, pred)
    active = np.abs(y) > epsilon_kw
    out["n_active_test"] = int(active.sum())
    out["active_fraction_test"] = float(active.mean()) if len(active) else None
    if active.any():
        active_metrics = _metrics(y[active], pred[active])
        out["active_only_mae_kw"] = active_metrics["mae_kw"]
        out["active_only_rmse_kw"] = active_metrics["rmse_kw"]
        out["active_only_r2"] = active_metrics["r2"]
    else:
        out["active_only_mae_kw"] = None
        out["active_only_rmse_kw"] = None
        out["active_only_r2"] = None
    return out


def run_walkforward(
    aligned_csv: str | Path,
    output_dir: str | Path,
    initial_train_fraction: float = 0.30,
    test_fraction: float = 0.10,
    step_fraction: float = 0.10,
    alpha: float = 10.0,
    activity_epsilon_kw: float = 0.05,
) -> dict[str, Any]:
    if not 0.1 <= initial_train_fraction < 0.9:
        raise ValueError("initial_train_fraction must be in [0.1, 0.9).")
    if not 0.01 <= test_fraction < 0.5:
        raise ValueError("test_fraction must be in [0.01, 0.5).")
    if not 0.01 <= step_fraction <= 0.5:
        raise ValueError("step_fraction must be in [0.01, 0.5].")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if activity_epsilon_kw < 0:
        raise ValueError("activity_epsilon_kw must be nonnegative.")

    path = Path(aligned_csv)
    if not path.exists():
        raise FileNotFoundError(f"Aligned HIPE CSV not found: {path}")
    df = pd.read_csv(path)
    if "timestamp_utc" not in df or "MainTerminal" not in df:
        raise ValueError("Aligned CSV must contain timestamp_utc and MainTerminal.")
    machines = [c for c in df.columns if c not in {"timestamp_utc", "MainTerminal"}]
    if not machines:
        raise ValueError("No machine reference columns found.")

    features = _features(df)
    x = features.to_numpy(dtype=float)
    valid_x = features.notna().all(axis=1).to_numpy()
    n = len(df)
    initial_n = int(n * initial_train_fraction)
    test_n = max(1, int(n * test_fraction))
    step_n = max(1, int(n * step_fraction))
    rows: list[dict[str, Any]] = []
    fold_num = 0
    train_end = initial_n

    while train_end < n:
        test_start = train_end
        test_end = min(n, test_start + test_n)
        fold_num += 1
        train_pos = np.arange(n) < train_end
        test_pos = (np.arange(n) >= test_start) & (np.arange(n) < test_end)
        for machine in machines:
            y = pd.to_numeric(df[machine], errors="coerce").to_numpy(dtype=float)
            tr = train_pos & valid_x & np.isfinite(y)
            te = test_pos & valid_x & np.isfinite(y)
            if tr.sum() < 100 or te.sum() < 20:
                rows.append({
                    "fold": fold_num, "machine": machine,
                    "train_start_index": 0, "train_end_index_exclusive": train_end,
                    "test_start_index": test_start, "test_end_index_exclusive": test_end,
                    "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                    "status": "skipped_insufficient_valid_rows",
                })
                continue
            ytr, yte = y[tr], y[te]
            pred_ridge = np.maximum(_fit_ridge_predict(x[tr], ytr, x[te], alpha), 0.0)
            pred_zero = np.zeros_like(yte)
            pred_mean = np.full_like(yte, max(float(np.mean(ytr)), 0.0))
            for model_name, pred in (
                ("ridge", pred_ridge),
                ("zero_baseline", pred_zero),
                ("train_mean_baseline", pred_mean),
            ):
                m = _metric_row(yte, pred, activity_epsilon_kw)
                rows.append({
                    "fold": fold_num,
                    "machine": machine,
                    "model": model_name,
                    "status": "evaluated",
                    "train_start_index": 0,
                    "train_end_index_exclusive": train_end,
                    "test_start_index": test_start,
                    "test_end_index_exclusive": test_end,
                    "train_start_timestamp_utc": str(df["timestamp_utc"].iloc[0]),
                    "train_end_timestamp_utc": str(df["timestamp_utc"].iloc[train_end - 1]),
                    "test_start_timestamp_utc": str(df["timestamp_utc"].iloc[test_start]),
                    "test_end_timestamp_utc": str(df["timestamp_utc"].iloc[test_end - 1]),
                    "train_rows": int(tr.sum()),
                    "test_rows": int(te.sum()),
                    "train_active_fraction": float(np.mean(np.abs(ytr) > activity_epsilon_kw)),
                    "alpha": float(alpha),
                    "activity_epsilon_kw": float(activity_epsilon_kw),
                    **m,
                })
        if test_end == n:
            break
        train_end += step_n

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "hipe_walkforward_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = {
        "data_type": "REAL_PUBLIC_DATASET_EVALUATION",
        "dataset": "HIPE one-week archive",
        "method": "Expanding-window chronological evaluation; each test block occurs after its training data.",
        "input_policy": "Only MainTerminal power history/rolling features and timestamp features are model inputs.",
        "target_policy": "Machine-level power is used only as supervised labels and evaluation reference, never as input features.",
        "total_aligned_rows": int(n),
        "initial_train_fraction": initial_train_fraction,
        "test_fraction": test_fraction,
        "step_fraction": step_fraction,
        "alpha": alpha,
        "activity_epsilon_kw": activity_epsilon_kw,
        "folds_evaluated": int(fold_num),
        "metrics_csv": str(metrics_path),
        "interpretation_notes": [
            "Inspect each fold's active_fraction_test; pooled averages can hide temporal distribution shift.",
            "active_only metrics condition on reference target activity and are diagnostic, not an online inference input.",
            "The activity threshold is heuristic, not official machine ON/OFF ground truth.",
            "A fold with no active target readings cannot establish active-load inference quality.",
            "Compare Ridge with zero-output and train-mean baselines in every fold.",
        ],
        "limitations": [
            "One public factory dataset and one week do not establish cross-factory generalization.",
            "Machine and MainTerminal readings are asynchronous; binning does not guarantee exact simultaneity.",
            "No energy savings or production scheduling improvements are measured.",
        ],
    }
    summary_path = outdir / "hipe_walkforward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_json": str(summary_path), "metrics_csv": str(metrics_path), "summary": summary}
