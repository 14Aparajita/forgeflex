from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from forgeflex.dashboard_utils import read_csv, read_json, numeric_columns
from forgeflex.scheduling import run_scheduling_simulation

ROOT = Path.cwd()
DEFAULT_REPORT = ROOT / "outputs/decision_report_v1/decision_report.json"
DEFAULT_HIPE = ROOT / "outputs/hipe/hipe_power_10s.csv"
DEFAULT_SENSOR = ROOT / "outputs/sensor_placement_v1/sensor_placement_recommendations.csv"
DEFAULT_METRICS = ROOT / "outputs/hipe_walkforward/hipe_walkforward_metrics.csv"

st.set_page_config(page_title="ForgeFlex | Energy Decision Demo", page_icon="⚡", layout="wide")
st.title("ForgeFlex")
st.caption("Adaptive-sensing production energy orchestrator · Interactive prototype")
st.warning(
    "Evidence boundary: HIPE is a public historical dataset; sensor priorities are retrospective heuristics; "
    "the scheduler uses simulated jobs and an illustrative tariff. Do not interpret simulated changes as measured factory savings."
)

with st.sidebar:
    st.header("Data sources")
    report_path = st.text_input("Decision report JSON", str(DEFAULT_REPORT))
    hipe_path = st.text_input("Aligned HIPE power CSV", str(DEFAULT_HIPE))
    sensor_path = st.text_input("Sensor recommendations CSV", str(DEFAULT_SENSOR))
    metrics_path = st.text_input("Walk-forward metrics CSV", str(DEFAULT_METRICS))
    st.divider()
    st.header("Scheduling scenario")
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
    job_count = st.slider("Synthetic jobs", min_value=5, max_value=60, value=20, step=5)
    horizon = st.slider("Scheduling horizon (slots)", min_value=8, max_value=48, value=24, step=4)
    slot_hours = st.select_slider("Hours per slot", options=[0.25, 0.5, 1.0, 2.0], value=1.0)
    run_scenario = st.button("Run scheduling scenario", type="primary", use_container_width=True)

report = read_json(report_path)
hipe = read_csv(hipe_path)
sensors = read_csv(sensor_path)
metrics = read_csv(metrics_path)

# Summary KPIs are derived from the saved report rather than hardcoded demo claims.
if report:
    evidence = report.get("evidence_types", {})
    hipe_summary = report.get("hipe", {})
    sensor_summary = report.get("sensor_placement", {})
    schedule = report.get("scheduling", {})
    cols = st.columns(4)
    cols[0].metric("Aligned HIPE rows", f"{len(hipe):,}" if not hipe.empty else "Unavailable")
    cols[1].metric("Power series", str(max(0, len(hipe.columns) - 1)) if not hipe.empty else "Unavailable")
    selected = sensor_summary.get("selected_for_submeter_validation", []) if isinstance(sensor_summary, dict) else []
    cols[2].metric("Sensor validation candidates", str(len(selected)) if isinstance(selected, list) else "See report")
    sched_delta = schedule.get("percent_change_optimized_vs_baseline", {}).get("energy_cost") if isinstance(schedule, dict) else None
    cols[3].metric("Simulated cost change", f"{sched_delta:+.2f}%" if isinstance(sched_delta, (int, float)) else "See report")
else:
    st.info("Decision report JSON was not found or could not be parsed. Set the correct path in the sidebar; other tabs can still be used where source files exist.")

power_tab, sensing_tab, scheduling_tab, validation_tab = st.tabs(
    ["Power intelligence", "Sensor priorities", "Scheduling sandbox", "Model validation"]
)

with power_tab:
    st.subheader("Historical power profile")
    if hipe.empty:
        st.info(f"No readable aligned power CSV at: {hipe_path}")
    else:
        st.caption("Public HIPE data, aligned to 10-second bins. This is not a live factory feed.")
        time_cols = [c for c in hipe.columns if any(k in c.lower() for k in ("time", "date", "timestamp"))]
        numeric = numeric_columns(hipe, set(time_cols))
        if not numeric:
            st.warning("The aligned file has no numeric columns available to plot.")
        else:
            chosen = st.multiselect("Power series to display", numeric, default=numeric[: min(4, len(numeric))])
            limit = st.slider("Rows to plot", min_value=100, max_value=max(100, min(len(hipe), 10000)), value=min(1000, max(100, min(len(hipe), 10000))), step=100)
            if chosen:
                plot_frame = hipe.loc[:, chosen].head(limit).copy()
                plot_frame.index.name = "sample"
                st.line_chart(plot_frame)
            st.dataframe(hipe.head(20), use_container_width=True)

with sensing_tab:
    st.subheader("Machines to validate with submeters")
    st.caption("These are retrospective validation priorities, not automatic installation decisions or ROI estimates.")
    if sensors.empty:
        st.info(f"No readable sensor recommendations at: {sensor_path}")
    else:
        st.dataframe(sensors, use_container_width=True, hide_index=True)
    if report:
        raw = report.get("sensor_placement", {})
        if isinstance(raw, dict):
            st.json(raw)

with scheduling_tab:
    st.subheader("Compare baseline and tariff-aware schedules")
    st.write("Change the scenario controls in the sidebar and run a new simulation. Both methods use the same generated jobs and illustrative tariff.")
    if run_scenario:
        out_dir = ROOT / "outputs" / "dashboard_scenarios" / f"seed_{int(seed)}_jobs_{int(job_count)}_horizon_{int(horizon)}"
        try:
            result = run_scheduling_simulation(
                output_dir=out_dir,
                seed=int(seed),
                job_count=int(job_count),
                horizon_slots=int(horizon),
                slot_hours=float(slot_hours),
            )
            st.session_state["forgeflex_latest_scenario"] = {"summary": result, "output_dir": str(out_dir)}
        except Exception as exc:
            st.error(f"Scenario failed: {type(exc).__name__}: {exc}")
    latest = st.session_state.get("forgeflex_latest_scenario")
    if latest:
        summary = latest["summary"]
        baseline = summary.get("baseline", {})
        optimized = summary.get("optimized", {})
        delta = summary.get("delta_optimized_minus_baseline", {})
        pct = summary.get("percent_change_optimized_vs_baseline", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline cost", f"{baseline.get('total_energy_cost', float('nan')):,.2f}")
        c2.metric("Tariff-aware cost", f"{optimized.get('total_energy_cost', float('nan')):,.2f}", f"{pct.get('energy_cost', 0):+.2f}%")
        c3.metric("Energy (baseline → optimized)", f"{baseline.get('total_energy_kwh', 0):,.2f} → {optimized.get('total_energy_kwh', 0):,.2f} kWh")
        c4.metric("Deadline misses", f"{baseline.get('deadline_misses_among_scheduled_jobs', 0)} → {optimized.get('deadline_misses_among_scheduled_jobs', 0)}")
        st.caption(f"Data type: {summary.get('data_type', 'UNKNOWN')} · Jobs: {summary.get('jobs_input_source', 'UNKNOWN')} · Tariff: {summary.get('tariff_input_source', 'UNKNOWN')}")
        st.json({"delta_optimized_minus_baseline": delta, "percent_change_optimized_vs_baseline": pct})
        base_path = Path(latest["output_dir"]) / "schedule_baseline.csv"
        opt_path = Path(latest["output_dir"]) / "schedule_optimized.csv"
        prof_path = Path(latest["output_dir"]) / "schedule_slot_profiles.csv"
        left, right = st.columns(2)
        with left:
            st.markdown("**Baseline schedule**")
            st.dataframe(read_csv(base_path), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Tariff-aware schedule**")
            st.dataframe(read_csv(opt_path), use_container_width=True, hide_index=True)
        profile = read_csv(prof_path)
        if not profile.empty:
            plot_cols = [c for c in profile.columns if c in ("aggregate_power_kw_baseline", "aggregate_power_kw_optimized")]
            if plot_cols:
                st.markdown("**Scheduled aggregate demand by slot**")
                st.line_chart(profile.set_index("slot")[plot_cols])
        st.download_button("Download scenario summary JSON", json.dumps(summary, indent=2), file_name="forgeflex_scenario_summary.json", mime="application/json")
        st.info("Simulation only: a lower tariff bill is not the same as lower energy consumption. Verify throughput and quality in a real deployment.")
    else:
        st.info("Set scenario inputs in the sidebar and click 'Run scheduling scenario' to generate a comparison.")

with validation_tab:
    st.subheader("Chronological model validation")
    if metrics.empty:
        st.info(f"No readable walk-forward metrics CSV at: {metrics_path}")
    else:
        st.caption("Fold-level held-out metrics. Aggregation and interpretation should respect each metric's definition and active-only conditioning.")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        numeric = numeric_columns(metrics)
        if numeric:
            possible_model_cols = [c for c in metrics.columns if any(t in c.lower() for t in ("model", "method", "baseline"))]
            metric_col = st.selectbox("Numeric metric to inspect", numeric)
            if possible_model_cols:
                st.bar_chart(metrics.set_index(possible_model_cols[0])[metric_col])

st.divider()
st.caption("ForgeFlex prototype · Local processing · Historical evidence and synthetic simulation are explicitly separated.")
