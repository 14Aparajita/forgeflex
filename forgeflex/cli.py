import argparse
import json

from .baseline import run_synthetic_baseline
from .activity_audit import run_activity_audit
from .config import SimulationConfig
from .hipe import load_hipe_power
from .inference import run_inference
from .walkforward import run_walkforward
from .two_stage import run_two_stage_walkforward
from .sensor_placement import run_sensor_placement
from .scheduling import run_scheduling_simulation
from .decision_report import run_decision_report
from .experiments import run_scheduling_experiments
from .constraint_aware import run_constraint_aware_schedule


def main():
    parser = argparse.ArgumentParser(prog="forgeflex")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="generate synthetic factory data and aggregate KPIs")
    p.add_argument("--output-dir", default="outputs/demo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=1440)
    p.add_argument("--interval-seconds", type=int, default=60)
    p.add_argument("--noise-std-kw", type=float, default=1.5)

    h = sub.add_parser("hipe", help="audit and align HIPE CSV power readings")
    h.add_argument("--data-dir", default="data/raw/hipe_week")
    h.add_argument("--output-dir", default="outputs/hipe")
    h.add_argument("--frequency", default="10s", help="Pandas resampling frequency, e.g. 10s or 30s")

    a = sub.add_parser("activity-audit", help="audit daily activity in raw and aligned HIPE data")
    a.add_argument("--data-dir", default="data/raw/hipe_week")
    a.add_argument("--aligned", default="outputs/hipe/hipe_power_10s.csv")
    a.add_argument("--output-dir", default="outputs/hipe_activity_audit")
    a.add_argument("--epsilon-kw", type=float, default=0.05, help="heuristic activity threshold; default 0.05 kW")

    inf = sub.add_parser("infer", help="fit and evaluate aggregate-only machine-power inference baseline")
    inf.add_argument("--input", default="outputs/hipe/hipe_power_10s.csv")
    inf.add_argument("--output-dir", default="outputs/inference")
    inf.add_argument("--train-fraction", type=float, default=0.70)
    inf.add_argument("--alpha", type=float, default=10.0)

    wf = sub.add_parser("walkforward", help="evaluate aggregate-only inference across expanding chronological windows")
    wf.add_argument("--input", default="outputs/hipe/hipe_power_10s.csv")
    wf.add_argument("--output-dir", default="outputs/hipe_walkforward")
    wf.add_argument("--initial-train-fraction", type=float, default=0.30)
    wf.add_argument("--test-fraction", type=float, default=0.10)
    wf.add_argument("--step-fraction", type=float, default=0.10)
    wf.add_argument("--alpha", type=float, default=10.0)
    wf.add_argument("--epsilon-kw", type=float, default=0.05)

    ts = sub.add_parser("two-stage", help="compare two-stage activity plus power inference with baselines in chronological folds")
    ts.add_argument("--input", default="outputs/hipe/hipe_power_10s.csv")
    ts.add_argument("--output-dir", default="outputs/hipe_two_stage")
    ts.add_argument("--initial-train-fraction", type=float, default=0.30)
    ts.add_argument("--test-fraction", type=float, default=0.10)
    ts.add_argument("--step-fraction", type=float, default=0.10)
    ts.add_argument("--alpha", type=float, default=10.0)
    ts.add_argument("--epsilon-kw", type=float, default=0.05)
    ts.add_argument("--activity-threshold", type=float, default=0.5)

    sp = sub.add_parser(
        "sensor-placement",
        help="rank candidate submeters from chronological inference validation metrics",
    )
    sp.add_argument("--metrics", default="outputs/hipe_walkforward/hipe_walkforward_metrics.csv")
    sp.add_argument("--output-dir", default="outputs/sensor_placement")
    sp.add_argument("--budget", type=int, default=3, help="maximum candidates to recommend for submeter validation")
    sp.add_argument("--min-active-samples", type=int, default=100)
    sp.add_argument("--min-active-folds", type=int, default=2)

    dr = sub.add_parser(
        "decision-report",
        help="combine existing HIPE, validation, sensor-priority, and scheduling artifacts into a report",
    )
    dr.add_argument("--output-dir", default="outputs/decision_report")
    dr.add_argument("--hipe-audit", default="outputs/hipe/hipe_audit.json")
    dr.add_argument("--activity-audit", default="outputs/hipe_activity_audit/hipe_activity_audit.json")
    dr.add_argument("--walkforward-metrics", default="outputs/hipe_walkforward/hipe_walkforward_metrics.csv")
    dr.add_argument("--sensor-summary", default="outputs/sensor_placement_v1/sensor_placement_summary.json")
    dr.add_argument("--sensor-recommendations", default="outputs/sensor_placement_v1/sensor_placement_recommendations.csv")
    dr.add_argument("--scheduling-summary", default="outputs/scheduling_demo_v1/scheduling_summary.json")

    sch = sub.add_parser(
        "schedule",
        help="compare earliest-available and tariff-aware production schedules in simulation",
    )
    sch.add_argument("--jobs", default=None, help="optional jobs CSV; otherwise generate a synthetic demo scenario")
    sch.add_argument("--tariff", default=None, help="optional tariff CSV with slot,price_per_kwh; otherwise use an illustrative synthetic tariff")
    sch.add_argument("--output-dir", default="outputs/scheduling_demo")
    sch.add_argument("--seed", type=int, default=42)
    sch.add_argument("--job-count", type=int, default=20)
    sch.add_argument("--horizon-slots", type=int, default=24)
    sch.add_argument("--slot-hours", type=float, default=1.0)

    ex = sub.add_parser(
        "experiments",
        help="run paired multi-scenario scheduling experiments (simulation only)",
    )
    ex.add_argument("--output-dir", default="outputs/scheduling_experiments_v1")
    ex.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 21])
    ex.add_argument("--job-counts", type=int, nargs="+", default=[10, 20, 30])
    ex.add_argument("--horizon-slots", type=int, default=24)
    ex.add_argument("--slot-hours", type=float, default=1.0)

    ca = sub.add_parser(
        "constraint-aware-schedule",
        help="compare baseline with conservative deadline-preserving tariff-aware rescheduling (simulation only)",
    )
    ca.add_argument("--jobs", default=None, help="optional jobs CSV; otherwise generate synthetic jobs")
    ca.add_argument("--tariff", default=None, help="optional tariff CSV with slot,price_per_kwh; otherwise use illustrative tariff")
    ca.add_argument("--output-dir", default="outputs/constraint_aware_demo")
    ca.add_argument("--seed", type=int, default=42)
    ca.add_argument("--job-count", type=int, default=20)
    ca.add_argument("--horizon-slots", type=int, default=24)
    ca.add_argument("--slot-hours", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "baseline":
        cfg = SimulationConfig(args.seed, args.samples, args.interval_seconds, args.noise_std_kw)
        print(json.dumps(run_synthetic_baseline(args.output_dir, cfg), indent=2))
    elif args.command == "hipe":
        result = load_hipe_power(args.data_dir, args.output_dir, args.frequency)
        print(json.dumps({
            "audit_json": result["audit_json"],
            "aligned_csv": result["aligned_csv"],
            "rows_after_alignment": result["summary"]["rows_after_alignment"],
            "columns": result["summary"]["columns"],
            "missing_fraction_by_series": result["summary"]["missing_fraction_by_series"],
            "data_type": result["summary"]["data_type"],
        }, indent=2))
    elif args.command == "activity-audit":
        result = run_activity_audit(args.data_dir, args.aligned, args.output_dir, args.epsilon_kw)
        print(json.dumps(result, indent=2))
    elif args.command == "walkforward":
        result = run_walkforward(
            args.input, args.output_dir, args.initial_train_fraction,
            args.test_fraction, args.step_fraction, args.alpha, args.epsilon_kw,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "two-stage":
        result = run_two_stage_walkforward(
            args.input, args.output_dir, args.initial_train_fraction,
            args.test_fraction, args.step_fraction, args.alpha,
            args.epsilon_kw, args.activity_threshold,
        )
        print(json.dumps({
            "summary_json": result["summary_json"],
            "metrics_csv": result["metrics_csv"],
            "folds_evaluated": result["summary"]["folds_evaluated"],
            "models": result["summary"]["models"],
            "data_type": result["summary"]["data_type"],
        }, indent=2))
    elif args.command == "infer":
        result = run_inference(args.input, args.output_dir, args.train_fraction, args.alpha)
        print(json.dumps({
            "metrics_json": result["metrics_json"],
            "predictions_csv": result["predictions_csv"],
            "data_type": result["summary"]["data_type"],
            "split_policy": result["summary"]["split_policy"],
            "metrics_by_machine": result["summary"]["metrics_by_machine"],
            "limitations": result["summary"]["limitations"],
        }, indent=2))
    elif args.command == "sensor-placement":
        result = run_sensor_placement(
            args.metrics, args.output_dir, args.budget,
            args.min_active_samples, args.min_active_folds,
        )
        print(json.dumps({
            "recommendations_csv": result["recommendations_csv"],
            "summary_json": result["summary_json"],
            "machines_seen": result["summary"]["machines_seen"],
            "eligible_machines": result["summary"]["eligible_machines"],
            "selected_for_submeter_validation": result["summary"]["selected_for_submeter_validation"],
            "data_type": result["summary"]["data_type"],
        }, indent=2))

    elif args.command == "decision-report":
        result = run_decision_report(
            output_dir=args.output_dir,
            hipe_audit_json=args.hipe_audit,
            activity_audit_json=args.activity_audit,
            walkforward_metrics_csv=args.walkforward_metrics,
            sensor_summary_json=args.sensor_summary,
            sensor_recommendations_csv=args.sensor_recommendations,
            scheduling_summary_json=args.scheduling_summary,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "experiments":
        result = run_scheduling_experiments(
            output_dir=args.output_dir,
            seeds=args.seeds,
            job_counts=args.job_counts,
            horizon_slots=args.horizon_slots,
            slot_hours=args.slot_hours,
        )
        print(json.dumps({
            "summary_json": result["summary_json"],
            "results_csv": result["results_csv"],
            "scenario_count": result["summary"]["scenario_count"],
            "data_type": result["summary"]["data_type"],
            "aggregate_by_tariff_condition": result["summary"]["aggregate_by_tariff_condition"],
        }, indent=2))
    elif args.command == "constraint-aware-schedule":
        result = run_constraint_aware_schedule(
            output_dir=args.output_dir,
            jobs_csv=args.jobs,
            tariff_csv=args.tariff,
            seed=args.seed,
            job_count=args.job_count,
            horizon_slots=args.horizon_slots,
            slot_hours=args.slot_hours,
        )
        summary = result["summary"]
        print(json.dumps({
            "summary_json": result["summary_json"],
            "baseline_schedule_csv": result["baseline_schedule_csv"],
            "constraint_aware_schedule_csv": result["constraint_aware_schedule_csv"],
            "data_type": summary["data_type"],
            "baseline_energy_cost": summary["baseline"]["total_energy_cost"],
            "constraint_aware_energy_cost": summary["constraint_aware"]["total_energy_cost"],
            "cost_reduction_percent_vs_baseline": summary["cost_reduction_percent_vs_baseline"],
            "guarantees_checked": summary["guarantees_checked"],
        }, indent=2))
    elif args.command == "schedule":
        result = run_scheduling_simulation(
            output_dir=args.output_dir,
            jobs_csv=args.jobs,
            tariff_csv=args.tariff,
            seed=args.seed,
            job_count=args.job_count,
            horizon_slots=args.horizon_slots,
            slot_hours=args.slot_hours,
        )
        print(json.dumps({
            "summary_json": result["summary_json"],
            "baseline_schedule_csv": result["baseline_schedule_csv"],
            "optimized_schedule_csv": result["optimized_schedule_csv"],
            "slot_profiles_csv": result["slot_profiles_csv"],
            "data_type": result["summary"]["data_type"],
            "baseline_energy_cost": result["summary"]["baseline"]["total_energy_cost"],
            "optimized_energy_cost": result["summary"]["optimized"]["total_energy_cost"],
            "baseline_energy_kwh": result["summary"]["baseline"]["total_energy_kwh"],
            "optimized_energy_kwh": result["summary"]["optimized"]["total_energy_kwh"],
            "baseline_deadline_misses": result["summary"]["baseline"]["deadline_misses_among_scheduled_jobs"],
            "optimized_deadline_misses": result["summary"]["optimized"]["deadline_misses_among_scheduled_jobs"],
        }, indent=2))


if __name__ == "__main__":
    main()
