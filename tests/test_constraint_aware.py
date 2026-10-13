import pandas as pd
from forgeflex.constraint_aware import run_constraint_aware_schedule


def test_constraint_aware_preserves_baseline_deadlines_and_job_count(tmp_path):
    jobs = pd.DataFrame([
        {"job_id":"A","machine":"M1","duration_slots":2,"power_kw":2.0,"release_slot":0,"due_slot":8,"quantity_units":3},
        {"job_id":"B","machine":"M1","duration_slots":2,"power_kw":1.0,"release_slot":0,"due_slot":10,"quantity_units":4},
        {"job_id":"C","machine":"M2","duration_slots":1,"power_kw":3.0,"release_slot":0,"due_slot":6,"quantity_units":2},
    ])
    tariff = pd.DataFrame({"slot":range(12),"price_per_kwh":[9,9,8,8,7,7,2,2,2,3,4,5]})
    jp, tp = tmp_path/"jobs.csv", tmp_path/"tariff.csv"
    jobs.to_csv(jp,index=False); tariff.to_csv(tp,index=False)
    result=run_constraint_aware_schedule(tmp_path/"out",str(jp),str(tp),horizon_slots=12,slot_hours=1.0)
    s=result["summary"]
    assert s["guarantees_checked"]["deadline_misses_not_worse_than_baseline"]
    assert s["guarantees_checked"]["scheduled_job_count_preserved"]
    assert s["guarantees_checked"]["per_machine_non_overlap"]
    base=pd.read_csv(result["baseline_schedule_csv"])
    opt=pd.read_csv(result["constraint_aware_schedule_csv"])
    assert int(base.scheduled.sum()) == int(opt.scheduled.sum())


def test_constraint_aware_does_not_claim_real_world_results(tmp_path):
    result=run_constraint_aware_schedule(tmp_path/"out",seed=42,job_count=8,horizon_slots=24)
    assert result["summary"]["data_type"] == "SIMULATION_ONLY"
