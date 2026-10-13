"""Aggregate-only inference baseline with diagnostic and naive comparisons.

Machine-level power is used only as supervised labels and evaluation targets.
Features use MainTerminal power history/current bin and timestamp features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_utc" not in frame.columns or "MainTerminal" not in frame.columns:
        raise ValueError("Input must contain timestamp_utc and MainTerminal columns.")
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    agg = pd.to_numeric(frame["MainTerminal"], errors="coerce")
    feats = pd.DataFrame(index=frame.index)
    feats["aggregate_now_kw"] = agg
    for lag in (1, 2, 3, 6, 18):
        feats[f"aggregate_lag_{lag}_kw"] = agg.shift(lag)
    feats["aggregate_roll_3_kw"] = agg.rolling(3, min_periods=3).mean()
    feats["aggregate_roll_18_kw"] = agg.rolling(18, min_periods=18).mean()
    seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    feats["time_day_sin"] = np.sin(2 * np.pi * seconds / 86400.0)
    feats["time_day_cos"] = np.cos(2 * np.pi * seconds / 86400.0)
    week_seconds = ts.dt.dayofweek * 86400 + seconds
    feats["time_week_sin"] = np.sin(2 * np.pi * week_seconds / (7 * 86400.0))
    feats["time_week_cos"] = np.cos(2 * np.pi * week_seconds / (7 * 86400.0))
    return feats.replace([np.inf, -np.inf], np.nan)


def _fit_ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xtr = (x_train - mean) / scale
    xte = (x_test - mean) / scale
    xtr = np.column_stack([np.ones(len(xtr)), xtr])
    xte = np.column_stack([np.ones(len(xte)), xte])
    penalty = np.eye(xtr.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xtr.T @ xtr + penalty, xtr.T @ y_train)
    return xte @ coef


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    sum_abs = float(np.sum(np.abs(y_true)))
    sum_true = float(np.sum(y_true))
    centered_ss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(errors ** 2) / centered_ss) if centered_ss > 1e-12 else None
    return {
        "mae_kw": mae,
        "rmse_kw": rmse,
        "r2": r2,
        "normalized_disaggregation_error": float(np.sum(np.abs(errors)) / sum_abs) if sum_abs > 1e-8 else None,
        "signed_energy_relative_error": float(sum(y_pred) / sum_true - 1.0) if abs(sum_true) > 1e-8 else None,
    }


def _target_diagnostics(y_train: np.ndarray, y_test: np.ndarray, activity_epsilon_kw: float) -> dict[str, Any]:
    def one_split(y: np.ndarray) -> dict[str, Any]:
        finite = y[np.isfinite(y)]
        if len(finite) == 0:
            return {"n": 0}
        active = np.abs(finite) > activity_epsilon_kw
        return {
            "n": int(len(finite)),
            "min_kw": float(np.min(finite)),
            "max_kw": float(np.max(finite)),
            "mean_kw": float(np.mean(finite)),
            "std_kw": float(np.std(finite)),
            "nonzero_fraction": float(np.mean(np.abs(finite) > 1e-8)),
            "active_fraction_above_epsilon": float(np.mean(active)),
            "activity_epsilon_kw": float(activity_epsilon_kw),
        }
    return {"train": one_split(y_train), "test": one_split(y_test)}


def run_inference(
    aligned_csv: str | Path,
    output_dir: str | Path,
    train_fraction: float = 0.70,
    alpha: float = 10.0,
    activity_epsilon_kw: float = 0.05,
) -> dict[str, Any]:
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if activity_epsilon_kw < 0:
        raise ValueError("activity_epsilon_kw must be nonnegative.")

    input_path = Path(aligned_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Aligned HIPE CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    if "MainTerminal" not in df.columns or "timestamp_utc" not in df.columns:
        raise ValueError("Aligned CSV must contain timestamp_utc and MainTerminal.")
    machine_cols = [c for c in df.columns if c not in {"timestamp_utc", "MainTerminal"}]
    if not machine_cols:
        raise ValueError("No machine reference columns found in aligned CSV.")

    features = _features(df)
    split_at = int(len(df) * train_fraction)
    train_positions = np.arange(len(df)) < split_at
    test_positions = ~train_positions
    feature_valid = features.notna().all(axis=1).to_numpy()
    if feature_valid[train_positions].sum() < 100:
        raise ValueError("Too few valid training feature rows after lag construction.")
    if feature_valid[test_positions].sum() < 20:
        raise ValueError("Too few valid test feature rows after lag construction.")

    predictions = pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "MainTerminal": pd.to_numeric(df["MainTerminal"], errors="coerce"),
    })
    result_metrics: dict[str, Any] = {}
    x_all = features.to_numpy(dtype=float)

    for machine in machine_cols:
        y = pd.to_numeric(df[machine], errors="coerce").to_numpy(dtype=float)
        train_mask = train_positions & feature_valid & np.isfinite(y)
        test_mask = test_positions & feature_valid & np.isfinite(y)
        y_train = y[train_mask]
        y_test = y[test_mask]
        if train_mask.sum() < 100 or test_mask.sum() < 20:
            result_metrics[machine] = {
                "status": "skipped_insufficient_valid_rows",
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "target_diagnostics": _target_diagnostics(y_train, y_test, activity_epsilon_kw),
            }
            for suffix in ("ridge", "zero_baseline", "train_mean_baseline"):
                predictions[f"{machine}__{suffix}_kw"] = np.nan
            continue

        ridge = np.maximum(_fit_ridge_predict(x_all[train_mask], y_train, x_all[test_mask], alpha), 0.0)
        zero = np.zeros_like(y_test)
        train_mean = max(float(np.mean(y_train)), 0.0)
        mean_pred = np.full_like(y_test, train_mean)

        model_preds = {
            "ridge": ridge,
            "zero_baseline": zero,
            "train_mean_baseline": mean_pred,
        }
        metric_block: dict[str, Any] = {}
        for name, pred in model_preds.items():
            predictions[f"{machine}__{name}_kw"] = np.nan
            predictions.loc[test_mask, f"{machine}__{name}_kw"] = pred
            metric_block[name] = _metrics(y_test, pred)

        ridge_mae = metric_block["ridge"]["mae_kw"]
        zero_mae = metric_block["zero_baseline"]["mae_kw"]
        mean_mae = metric_block["train_mean_baseline"]["mae_kw"]
        metric_block["ridge_mae_vs_zero_baseline"] = (
            float(ridge_mae / zero_mae) if zero_mae > 1e-12 else None
        )
        metric_block["ridge_mae_vs_train_mean_baseline"] = (
            float(ridge_mae / mean_mae) if mean_mae > 1e-12 else None
        )
        result_metrics[machine] = {
            "status": "evaluated",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "target_diagnostics": _target_diagnostics(y_train, y_test, activity_epsilon_kw),
            "metrics": metric_block,
            "model": "nonnegative_clipped_ridge_regression",
            "alpha": float(alpha),
        }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "hipe_inference_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    summary = {
        "data_type": "REAL_PUBLIC_DATASET_EVALUATION",
        "dataset": "HIPE one-week archive",
        "input_policy": "Only MainTerminal power history/rolling features and timestamp features are model inputs.",
        "target_policy": "Machine-level power is used as supervised training labels and held-out reference values, never input features.",
        "split_policy": f"Chronological split: first {train_fraction:.0%} train, remaining {1-train_fraction:.0%} test; no random shuffle.",
        "split_index": split_at,
        "total_aligned_rows": int(len(df)),
        "feature_names": list(features.columns),
        "alpha": float(alpha),
        "activity_epsilon_kw": float(activity_epsilon_kw),
        "metrics_by_machine": result_metrics,
        "prediction_csv": str(predictions_path),
        "interpretation_notes": [
            "Always inspect target activity fractions before interpreting MAE or R².",
            "R² is undefined when the held-out target has zero variance.",
            "Normalized and signed energy errors are undefined when the held-out target's absolute/total energy denominator is approximately zero.",
            "A low error on a near-zero target is not evidence of useful machine-state inference.",
            "Compare Ridge against zero and train-mean baselines before claiming value.",
        ],
        "limitations": [
            "This is a simple baseline, not a validated production-grade NILM system.",
            "Machine and terminal readings are asynchronous; 10-second binning does not guarantee exact simultaneity.",
            "This is supervised disaggregation; it is not unsupervised inference.",
            "The test period is later in the same one-week dataset and does not establish cross-factory generalization.",
            "No energy savings or production scheduling improvements are measured.",
        ],
    }
    metrics_path = output / "hipe_inference_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"metrics_json": str(metrics_path), "predictions_csv": str(predictions_path), "summary": summary}
