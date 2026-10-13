import pandas as pd
import pytest
from forgeflex.metrics import aggregate_summary, percentage_reduction

def test_energy_kwh():
    assert aggregate_summary(pd.DataFrame({"aggregate_power_kw":[60.,60.]}),60)["energy_kwh"] == pytest.approx(2.)

def test_reduction_validation():
    assert percentage_reduction(100,80) == pytest.approx(20)
    with pytest.raises(ValueError):
        percentage_reduction(0,1)
