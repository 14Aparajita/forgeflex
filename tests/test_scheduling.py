import json

import pandas as pd
import pytest

from forgeflex.scheduling import run_scheduling_simulation


def _write_case(tmp_path, jobs, prices):
    jobs_path = tmp_path / "jobs.csv"
    tariff_path = tmp_path / "tariff.csv"
    pd.DataFrame(jobs).to_csv(jobs_path, index=False)
    pd.DataFrame({"slot": range(len(prices)), "price_per_kwh": prices}).to_csv(tariff_path, index=False)
    return jobs_path, tariff_path


def test_cost_aware_schedule_shifts_flexible_jobs_to_cheaper_slots(tmp_path):
    jobs_path, tariff_path = _write_case(
        tmp_path,
        [
            {"job_id": "A", "machine": "M1", "duration_slots": 1, "power_kw": 1, "release_slot": 0, "due_slot": 4, "quantity_units": 10},
            {"job_id": "B", "machine": "M1", "duration_slots": 1, "power_kw": 1, "release_slot": 0, "due_slot": 4, "quantity_units": 10},
        ],
        [10, 10, 1, 1],
    )
    result = run_scheduling_simulation(
        tmp_path / "out", jobs_csv=jobs_path, tariff_csv=tariff_path,
        horizon_slots=4, slot_hours=1.0,
    )
    assert result["summary"]["baseline"]["total_energy_cost"] == pytest.approx(20)
    assert result["summary"]["optimized"]["total_energy_cost"] == pytest.approx(2)
    assert result["summary"]["baseline"]["total_energy_kwh"] == pytest.approx(2)
    assert result["summary"]["optimized"]["total_energy_kwh"] == pytest.approx(2)
    assert result["summary"]["optimized"]["deadline_misses_among_scheduled_jobs"] == 0


def test_machine_jobs_never_overlap(tmp_path):
    jobs_path, tariff_path = _write_case(
        tmp_path,
        [
            {"job_id": "A", "machine": "M1", "duration_slots": 2, "power_kw": 2, "release_slot": 0, "due_slot": 6},
            {"job_id": "B", "machine": "M1", "duration_slots": 2, "power_kw": 3, "release_slot": 0, "due_slot": 6},
            {"job_id": "C", "machine": "M2", "duration_slots": 2, "power_kw": 1, "release_slot": 0, "due_slot": 6},
        ],
        [8, 8, 2, 2, 4, 4],
    )
    result = run_scheduling_simulation(tmp_path / "out", jobs_path, tariff_path, horizon_slots=6)
    schedule = pd.read_csv(result["optimized_schedule_csv"])
    for _, group in schedule[schedule.scheduled == True].groupby("machine"):
        intervals = sorted(zip(group.start_slot, group.completion_slot))
        assert all(intervals[i][1] <= intervals[i + 1][0] for i in range(len(intervals) - 1))


def test_default_run_labels_synthetic_inputs_and_outputs(tmp_path):
    result = run_scheduling_simulation(tmp_path / "out", seed=7, job_count=10, horizon_slots=24)
    summary = result["summary"]
    assert summary["data_type"] == "SIMULATION_ONLY"
    assert summary["jobs_input_source"] == "SYNTHETIC_DEMO_INPUT"
    assert summary["tariff_input_source"] == "SYNTHETIC_ILLUSTRATIVE_TARIFF"
    assert summary["baseline"]["jobs_requested"] == 10
    assert summary["optimized"]["jobs_requested"] == 10
    assert (tmp_path / "out" / "scheduling_summary.json").exists()
    saved = json.loads((tmp_path / "out" / "scheduling_summary.json").read_text(encoding="utf-8"))
    assert saved["data_type"] == "SIMULATION_ONLY"


def test_invalid_jobs_rejected(tmp_path):
    jobs_path, tariff_path = _write_case(
        tmp_path,
        [{"job_id": "A", "machine": "M1", "duration_slots": 0, "power_kw": 1, "release_slot": 0, "due_slot": 4}],
        [1, 1, 1, 1],
    )
    with pytest.raises(ValueError, match="duration_slots"):
        run_scheduling_simulation(tmp_path / "out", jobs_path, tariff_path, horizon_slots=4)


def test_tariff_requires_complete_slot_coverage(tmp_path):
    jobs_path = tmp_path / "jobs.csv"
    tariff_path = tmp_path / "tariff.csv"
    pd.DataFrame([{"job_id": "A", "machine": "M1", "duration_slots": 1, "power_kw": 1, "release_slot": 0, "due_slot": 3}]).to_csv(jobs_path, index=False)
    pd.DataFrame({"slot": [0, 1], "price_per_kwh": [1, 2]}).to_csv(tariff_path, index=False)
    with pytest.raises(ValueError, match="exactly slots"):
        run_scheduling_simulation(tmp_path / "out", jobs_path, tariff_path, horizon_slots=3)
