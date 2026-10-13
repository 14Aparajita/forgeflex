"""Retrospective sensor-placement prioritization from chronological validation.

This module ranks machines for *submeter validation* using out-of-sample
aggregate-only inference metrics. It does not estimate installation ROI or
energy savings. The output is a decision-support shortlist, not an automatic
installation instruction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "machine", "model", "status", "mae_kw", "test_rows",
    "n_active_test", "active_only_mae_kw",
}


def _as_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & (weights > 0) & np.isfinite(values)
    if not valid.any():
        return None
    v = values.loc[valid].to_numpy(dtype=float)
    w = weights.loc[valid].to_numpy(dtype=float)
    return float(np.average(v, weights=w))


def _fold_keys(frame: pd.DataFrame) -> list[str]:
    # Include interval boundaries when available so baseline comparisons use
    # the same chronological test windows, not merely similarly numbered folds.
    keys = ["machine"]
    for col in ("fold", "test_start_index", "test_end_index_exclusive"):
        if col in frame.columns:
            keys.append(col)
    return keys


def _baseline_disadvantage(
    metrics: pd.DataFrame,
    machine: str,
    ridge_weighted_mae: float | None,
) -> float | None:
    """Relative Ridge MAE excess over the best available naive baseline.

    Positive values mean Ridge is worse than at least one naive baseline.
    Zero means Ridge matches or beats the best naive baseline. This is a
    heuristic ranking feature, not a measure of causal sensor value.
    """
    if ridge_weighted_mae is None:
        return None
    group = metrics[
        (metrics["machine"].astype(str) == machine)
        & (metrics["status"].astype(str) == "evaluated")
        & metrics["model"].isin(["zero_baseline", "train_mean_baseline"])
    ].copy()
    if group.empty:
        return None
    group["_mae"] = _as_numeric(group, "mae_kw")
    group["_weight"] = _as_numeric(group, "test_rows")
    baseline_values = []
    for _, part in group.groupby("model"):
        value = _weighted_mean(part["_mae"], part["_weight"])
        if value is not None and np.isfinite(value):
            baseline_values.append(value)
    if not baseline_values:
        return None
    best_naive = min(baseline_values)
    return float(
        max(0.0, ridge_weighted_mae - best_naive)
        / max(best_naive, ridge_weighted_mae, 0.05)
    )


def _percentile_rank(values: pd.Series) -> pd.Series:
    """Map eligible candidate values to deterministic percentile ranks."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = numeric.notna() & np.isfinite(numeric)
    if valid.any():
        result.loc[valid] = numeric.loc[valid].rank(method="average", pct=True)
    return result


def run_sensor_placement(
    metrics_csv: str | Path,
    output_dir: str | Path,
    budget: int = 3,
    min_active_samples: int = 100,
    min_active_folds: int = 2,
) -> dict[str, Any]:
    """Create a sensor-priority shortlist from walk-forward Ridge metrics.

    Only evaluated out-of-sample metric rows are used. The ranking combines:
      * 50% percentile rank of active-only MAE (higher error => higher priority)
      * 30% percentile rank of active fraction (more active exposure => higher)
      * 20% percentile rank of Ridge's relative excess error vs naive baselines

    Candidates below the minimum active-evidence thresholds are not ranked.
    Selected rows are recommendations to *validate* a submeter candidate.
    """
    if budget < 0:
        raise ValueError("budget must be >= 0.")
    if min_active_samples < 1:
        raise ValueError("min_active_samples must be >= 1.")
    if min_active_folds < 1:
        raise ValueError("min_active_folds must be >= 1.")

    input_path = Path(metrics_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Walk-forward metrics CSV not found: {input_path}")

    metrics = pd.read_csv(input_path)
    missing = sorted(_REQUIRED_COLUMNS - set(metrics.columns))
    if missing:
        raise ValueError(
            "Walk-forward metrics CSV is missing required columns: "
            + ", ".join(missing)
        )

    metrics["machine"] = metrics["machine"].astype(str)
    metrics["model"] = metrics["model"].astype(str)
    metrics["status"] = metrics["status"].astype(str)
    evaluated = metrics[
        (metrics["status"] == "evaluated")
        & metrics["model"].isin(["ridge", "zero_baseline", "train_mean_baseline"])
    ].copy()
    ridge = evaluated[evaluated["model"] == "ridge"].copy()
    if ridge.empty:
        raise ValueError("No evaluated Ridge rows found in the metrics CSV.")

    for column in ("mae_kw", "test_rows", "n_active_test", "active_only_mae_kw"):
        ridge[column] = _as_numeric(ridge, column)
    if "active_fraction_test" in ridge.columns:
        ridge["active_fraction_test"] = _as_numeric(ridge, "active_fraction_test")
    else:
        ridge["active_fraction_test"] = np.nan

    rows: list[dict[str, Any]] = []
    for machine, part in ridge.groupby("machine", sort=True):
        part = part.copy()
        total_test_rows = float(part["test_rows"].fillna(0).clip(lower=0).sum())
        total_active_samples = int(part["n_active_test"].fillna(0).clip(lower=0).sum())
        active_part = part[part["n_active_test"].fillna(0) > 0]
        active_fold_count = int((part["n_active_test"].fillna(0) > 0).sum())

        active_mae = _weighted_mean(
            active_part["active_only_mae_kw"],
            active_part["n_active_test"],
        )
        active_fraction = (
            float(total_active_samples / total_test_rows)
            if total_test_rows > 0 else None
        )
        ridge_mae = _weighted_mean(part["mae_kw"], part["test_rows"])
        disadvantage = _baseline_disadvantage(evaluated, machine, ridge_mae)

        eligible = (
            active_fold_count >= min_active_folds
            and total_active_samples >= min_active_samples
            and active_mae is not None
            and active_fraction is not None
        )
        reasons = []
        if active_fold_count < min_active_folds:
            reasons.append(
                f"active samples observed in {active_fold_count} validation folds; "
                f"minimum is {min_active_folds}"
            )
        if total_active_samples < min_active_samples:
            reasons.append(
                f"{total_active_samples} active test samples; minimum is {min_active_samples}"
            )
        if active_mae is None:
            reasons.append("active-only Ridge error is unavailable")

        rows.append({
            "machine": machine,
            "evaluated_ridge_folds": int(len(part)),
            "active_test_folds": active_fold_count,
            "test_rows_across_folds": int(total_test_rows),
            "active_test_samples_across_folds": total_active_samples,
            "active_fraction_across_folds": active_fraction,
            "ridge_mean_mae_kw_weighted_by_test_rows": ridge_mae,
            "ridge_active_only_mae_kw_weighted_by_active_samples": active_mae,
            "ridge_relative_excess_mae_vs_best_naive_baseline": disadvantage,
            "eligible_for_ranking": bool(eligible),
            "eligibility_note": "eligible" if eligible else "; ".join(reasons),
            "recommendation_status": (
                "pending" if eligible else "insufficient_validation_evidence"
            ),
            "active_error_percentile": None,
            "activity_percentile": None,
            "baseline_disadvantage_percentile": None,
            "priority_score_0_to_100": None,
            "priority_tier": "insufficient_evidence" if not eligible else "pending",
            "recommendation_rank": None,
            "recommendation": (
                "Validate a candidate submeter here if installation feasibility, "
                "machine criticality, and cost justify it."
                if eligible else
                "Collect more representative validation evidence before ranking this machine."
            ),
        })

    result = pd.DataFrame(rows)
    eligible_mask = result["eligible_for_ranking"]
    eligible_rows = result.loc[eligible_mask].copy()

    if not eligible_rows.empty:
        active_error_rank = _percentile_rank(
            eligible_rows["ridge_active_only_mae_kw_weighted_by_active_samples"]
        )
        activity_rank = _percentile_rank(
            eligible_rows["active_fraction_across_folds"]
        )
        baseline_rank = _percentile_rank(
            eligible_rows["ridge_relative_excess_mae_vs_best_naive_baseline"].fillna(0.0)
        )
        eligible_rows["active_error_percentile"] = active_error_rank.round(4)
        eligible_rows["activity_percentile"] = activity_rank.round(4)
        eligible_rows["baseline_disadvantage_percentile"] = baseline_rank.round(4)
        scores = 100.0 * (
            0.50 * active_error_rank
            + 0.30 * activity_rank
            + 0.20 * baseline_rank
        )
        eligible_rows["priority_score_0_to_100"] = scores.round(4)
        eligible_rows = eligible_rows.sort_values(
            ["priority_score_0_to_100", "ridge_active_only_mae_kw_weighted_by_active_samples", "machine"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        eligible_rows["recommendation_rank"] = np.arange(1, len(eligible_rows) + 1)
        eligible_rows["priority_tier"] = np.select(
            [eligible_rows["priority_score_0_to_100"] >= 67.0,
             eligible_rows["priority_score_0_to_100"] >= 34.0],
            ["high", "medium"],
            default="lower",
        )
        eligible_rows["recommendation_status"] = np.where(
            eligible_rows["recommendation_rank"] <= budget,
            "recommend_validate_submeter_candidate",
            "ranked_candidate_outside_budget",
        )
        selected = eligible_rows[eligible_rows["recommendation_rank"] <= budget]
        selected_names = set(selected["machine"].astype(str))
        for idx in eligible_rows.index:
            machine = str(eligible_rows.loc[idx, "machine"])
            tier = str(eligible_rows.loc[idx, "priority_tier"])
            score = float(eligible_rows.loc[idx, "priority_score_0_to_100"])
            if machine in selected_names:
                eligible_rows.loc[idx, "recommendation"] = (
                    f"Priority tier {tier}; score {score:.1f}/100. Validate this machine "
                    "as a submeter candidate, then check installation cost, criticality, "
                    "electrical feasibility, and expected decision value before installing."
                )
            else:
                eligible_rows.loc[idx, "recommendation"] = (
                    f"Priority tier {tier}; score {score:.1f}/100. Retain as a candidate "
                    "for a larger sensing budget or after business-priority review."
                )

        result.loc[eligible_rows.index, eligible_rows.columns] = eligible_rows

    result = result.sort_values(
        by=["eligible_for_ranking", "recommendation_rank", "machine"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    recommendations_path = outdir / "sensor_placement_recommendations.csv"
    result.to_csv(recommendations_path, index=False)

    selected_rows = result[
        result["recommendation_status"] == "recommend_validate_submeter_candidate"
    ]
    summary = {
        "data_type": "RETROSPECTIVE_VALIDATION_PRIORITIZATION",
        "input_metrics_csv": str(input_path),
        "method": (
            "Heuristic ranking from out-of-sample chronological Ridge validation metrics; "
            "prioritizes high active-only error, frequent activity, and Ridge underperformance "
            "relative to naive baselines."
        ),
        "score_definition": {
            "active_only_mae_percentile_weight": 0.50,
            "active_fraction_percentile_weight": 0.30,
            "ridge_excess_error_vs_best_naive_percentile_weight": 0.20,
            "scale": "0 to 100; relative percentile score among eligible machines only",
            "note": "A ranking heuristic, not a calibrated probability, ROI, or physical sensor-value estimate.",
        },
        "eligibility": {
            "minimum_active_test_samples_across_folds": min_active_samples,
            "minimum_folds_with_active_test_samples": min_active_folds,
        },
        "budget_slots": budget,
        "machines_seen": int(result["machine"].nunique()),
        "eligible_machines": int(result["eligible_for_ranking"].sum()),
        "selected_for_submeter_validation": selected_rows["machine"].astype(str).tolist(),
        "recommendations_csv": str(recommendations_path),
        "interpretation_notes": [
            "Recommendations mean validate a candidate submeter, not automatically install one.",
            "Metrics come from chronological held-out evaluation; active-only errors condition on reference target activity.",
            "The percentile score is relative to eligible machines in this dataset and changes if the candidate set changes.",
            "This version does not include installation costs, electrical feasibility, machine criticality, or independently verified energy share.",
            "No energy savings, ROI, payback period, or causal benefit from adding a sensor is estimated.",
            "HIPE is a public one-week dataset, not the user's factory; results do not establish cross-factory generalization.",
            "When test windows overlap, counts across folds may count the same timestamps more than once; use the default non-overlapping test/step fractions for straightforward interpretation.",
        ],
    }
    summary_path = outdir / "sensor_placement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "recommendations_csv": str(recommendations_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }
