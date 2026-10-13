import pandas as pd

def clean_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "aggregate_power_kw"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    result["aggregate_power_kw"] = pd.to_numeric(result["aggregate_power_kw"], errors="coerce")
    if result["timestamp"].isna().any() or result["aggregate_power_kw"].isna().any():
        raise ValueError("timestamp/power contains missing or invalid values")
    if not result["aggregate_power_kw"].map(lambda x: float("-inf") < x < float("inf")).all():
        raise ValueError("power contains infinite values")
    if (result["aggregate_power_kw"] < 0).any():
        raise ValueError("aggregate power must be non-negative after meter normalization")
    return result.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
