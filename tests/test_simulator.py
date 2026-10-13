import numpy as np
from forgeflex.config import SimulationConfig
from forgeflex.simulator import generate_factory_trace

def test_deterministic():
    a, b = generate_factory_trace(SimulationConfig(seed=7, samples=100))
    c, d = generate_factory_trace(SimulationConfig(seed=7, samples=100))
    assert a.equals(c) and b.equals(d)

def test_aggregate_sum_when_noise_zero():
    machines, aggregate = generate_factory_trace(SimulationConfig(seed=3, samples=200, noise_std_kw=0))
    expected = machines.drop(columns="timestamp").sum(axis=1).to_numpy()
    assert np.allclose(expected, aggregate["aggregate_power_kw"].to_numpy())
