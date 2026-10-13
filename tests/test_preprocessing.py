import pandas as pd
import pytest
from forgeflex.preprocessing import clean_aggregate

def test_sort_and_deduplicate():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01 00:01","2026-01-01 00:00","2026-01-01 00:01"]),
                          "aggregate_power_kw": [2.,1.,3.]})
    out = clean_aggregate(frame)
    assert len(out) == 2
    assert out.iloc[0]["aggregate_power_kw"] == 1.
    assert out.iloc[1]["aggregate_power_kw"] == 3.

def test_reject_negative():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"]), "aggregate_power_kw": [-1.]})
    with pytest.raises(ValueError):
        clean_aggregate(frame)
