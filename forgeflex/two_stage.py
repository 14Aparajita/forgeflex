"""Leakage-safe two-stage aggregate-only inference walk-forward experiment.

Stage 1 predicts heuristic activity labels from aggregate-meter history/time features.
Stage 2 estimates power from training examples labelled active. Test machine targets
are used only for evaluation, never as model inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import _features, _fit_ridge_predict, _metrics


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _spectral_norm_squared(matrix: np.ndarray, iterations: int = 20) -> float:
    """Approximate squared spectral norm using deterministic power iteration."""
    if matrix.size == 0:
        return 0.0
    vector = np.ones(matrix.shape[1], dtype=float)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    for _ in range(iterations):
        vector = matrix.T @ (matrix @ vector)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            return 0.0
        vector /= norm
    projected = matrix @ vector
    return float(np.dot(projected, projected))


def _fit_activity_classifier(
    x_train: np.ndarray,
    y_active: np.ndarray,
    x_test: np.ndarray,
    l2: float = 1.0,
    iterations: int = 500,
) -> np.ndarray:
    """Balanced logistic regression implemented with NumPy; returns probabilities."""
    if len(y_active) == 0:
        return np.zeros(len(x_test), dtype=float)
    if len(np.unique(y_active)) < 2:
        return np.full(len(x_test), float(y_active[0]))

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-10] = 1.0
    a = np.column_stack([np.ones(len(x_train)), (x_train - mean) / scale])
    b = np.column_stack([np.ones(len(x_test)), (x_test - mean) / scale])

    positives = max(int(y_active.sum()), 1)
    negatives = max(len(y_active) - positives, 1)
    weights = np.where(
        y_active > 0,
        len(y_active) / (2 * positives),
        len(y_active) / (2 * negatives),
    )
    coef = np.zeros(a.shape[1], dtype=float)
    penalty = np.ones(a.shape[1], dtype=float)
    penalty[0] = 0.0

    spectral = _spectral_norm_squared(a) / max(len(a), 1)
    lr = 1.0 / max(0.25 * spectral + l2, 1.0)
    for _ in range(iterations):
        probs = _sigmoid(a @ coef)
        grad = (
            a.T @ (weights * (probs - y_active)) / len(a)
            + l2 * penalty * coef / len(a)
        )
        coef -= lr * grad
    return _sigmoid(b @ coef)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = 0.0 if precision == 0 or recall == 0 else None
    elif precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "activity_precision": precision,
        "activity_recall": recall,
        "activity_f1": f1,
        "activity_true_positive": tp,
        "activity_false_positive": fp,
        "activity_false_negative": fn,
        "activity_true_negative": tn,
    }


def _fold_metric_row(y: np.ndarray, pred: np.ndarray, epsilon: float) -> dict[str, Any]:
    out = _metrics(y, pred)
    active = np.abs(y) > epsilon
    out["n_active_test"] = int(active.sum())
    out["active_fraction_test"] = float(active.mean()) if len(active) else None
    if active.any():
        active_metrics = _metrics(y[active], pred[active])
        out["active_only_mae_kw"] = active_metrics["mae_kw"]
        out["active_only_rmse_kw"] = active_metrics["rmse_kw"]
    else:
        out["active_only_mae_kw"] = None
        out["active_only_rmse_kw"] = None
    return out


def run_two_stage_walkforward(
    aligned_csv: str | Path,
    output_dir: str | Path,
    initial_train_fraction: float = 0.30,
    test_fraction: float = 0.10,
    step_fraction: float = 0.10,
    alpha: float = 10.0,
    activity_epsilon_kw: float = 0.05,
    activity_threshold: float = 0.5,
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
    if not 0.0 < activity_threshold < 1.0:
        raise ValueError("activity_threshold must be between 0 and 1.")

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
                    "fold": fold_num, "machine": machine, "model": "two_stage",
                    "status": "skipped_insufficient_valid_rows",
                    "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                })
                continue

            xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]
            y_active_tr = (np.abs(ytr) > activity_epsilon_kw).astype(float)
            y_active_te = np.abs(yte) > activity_epsilon_kw
            pred_ridge = np.maximum(_fit_ridge_predict(xtr, ytr, xte, alpha), 0.0)
            pred_zero = np.zeros_like(yte)
            pred_mean = np.full_like(yte, max(float(np.mean(ytr)), 0.0))

            probabilities = _fit_activity_classifier(xtr, y_active_tr, xte)
            predicted_active = probabilities >= activity_threshold
            active_rows = y_active_tr > 0
            if active_rows.sum() >= 20:
                active_power = np.maximum(
                    _fit_ridge_predict(
                        xtr[active_rows], np.maximum(ytr[active_rows], 0.0), xte, alpha
                    ),
                    0.0,
                )
            elif active_rows.any():
                active_power = np.full(
                    len(yte), max(float(np.mean(ytr[active_rows])), 0.0)
                )
            else:
                active_power = np.zeros(len(yte))
            pred_two_stage = np.where(predicted_active, active_power, 0.0)

            predictions = {
                "ridge": pred_ridge,
                "two_stage": pred_two_stage,
                "zero_baseline": pred_zero,
                "train_mean_baseline": pred_mean,
            }
            cls = _classification_metrics(y_active_te, predicted_active)
            diagnostics = {
                "predicted_active_fraction": float(np.mean(predicted_active)),
                "active_train_min_kw": float(np.min(ytr[active_rows])) if active_rows.any() else None,
                "active_train_max_kw": float(np.max(ytr[active_rows])) if active_rows.any() else None,
                "active_power_prediction_min_kw": float(np.min(active_power)) if len(active_power) else None,
                "active_power_prediction_max_kw": float(np.max(active_power)) if len(active_power) else None,
                "active_power_prediction_mean_kw": float(np.mean(active_power)) if len(active_power) else None,
                "final_prediction_max_kw": float(np.max(pred_two_stage)),
                "target_test_max_kw": float(np.max(yte)),
            }

            for model_name, pred in predictions.items():
                metric = _fold_metric_row(yte, pred, activity_epsilon_kw)
                row = {
                    "fold": fold_num, "machine": machine, "model": model_name,
                    "status": "evaluated", "train_start_index": 0,
                    "train_end_index_exclusive": train_end,
                    "test_start_index": test_start, "test_end_index_exclusive": test_end,
                    "train_start_timestamp_utc": str(df["timestamp_utc"].iloc[0]),
                    "train_end_timestamp_utc": str(df["timestamp_utc"].iloc[train_end - 1]),
                    "test_start_timestamp_utc": str(df["timestamp_utc"].iloc[test_start]),
                    "test_end_timestamp_utc": str(df["timestamp_utc"].iloc[test_end - 1]),
                    "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                    "train_active_fraction": float(y_active_tr.mean()),
                    "activity_threshold": float(activity_threshold),
                    "activity_epsilon_kw": float(activity_epsilon_kw),
                    "alpha": float(alpha), **metric,
                }
                if model_name == "two_stage":
                    row.update(cls)
                    row["mean_predicted_activity_probability"] = float(np.mean(probabilities))
                    row.update(diagnostics)
                rows.append(row)

        if test_end == n:
            break
        train_end += step_n

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "hipe_two_stage_walkforward_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    summary = {
        "data_type": "INPUT_DATASET_EVALUATION",
        "dataset": path.name,
        "method": "Expanding-window chronological evaluation; two-stage activity classifier and conditional power regressor trained only on each fold's training prefix.",
        "input_policy": "Only MainTerminal power history/rolling features and timestamp features are model inputs.",
        "target_policy": "Machine-level power is used for training labels and held-out evaluation only, never as input features.",
        "activity_label_policy": f"Activity label is abs(machine power) > {activity_epsilon_kw} kW; heuristic only, not official machine ON/OFF ground truth.",
        "models": ["ridge", "two_stage", "zero_baseline", "train_mean_baseline"],
        "activity_threshold": activity_threshold,
        "initial_train_fraction": initial_train_fraction,
        "test_fraction": test_fraction,
        "step_fraction": step_fraction,
        "alpha": alpha,
        "total_aligned_rows": int(n),
        "folds_evaluated": int(fold_num),
        "metrics_csv": str(metrics_path),
        "interpretation_notes": [
            "Compare models on identical fold/machine rows.",
            "Two-stage classifier uses only aggregate features; target activity labels are used only during training and evaluation.",
            "Activity-conditioned metrics use reference target values to select test rows and are diagnostic only.",
            "Inspect fold-level activity and classifier precision/recall; pooled metrics can hide temporal distribution shift.",
            "Prediction diagnostics do not establish model validity.",
            "No energy savings or cross-factory generalization is established by this experiment.",
        ],
        "limitations": [
            "One public factory dataset and one week only.",
            "Machine and MainTerminal readings are asynchronous.",
            "The activity threshold is heuristic and sensor readings may include noise or offsets.",
            "No energy savings or production scheduling improvements are measured.",
            "The activity classifier uses approximate spectral-norm estimation for its gradient step.",
        ],
    }
    summary_path = outdir / "hipe_two_stage_walkforward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary_json": str(summary_path), "metrics_csv": str(metrics_path), "summary": summary}
