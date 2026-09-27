"""
Latency percentile calculations. Implemented explicitly (not delegated to a
library) since this is a core, interview-explainable piece of the
observability story. Operates on real recorded latencies only — returns
None for percentiles that don't have enough samples rather than fabricating
a number.
"""
import math


def percentile(values: list[float], p: float) -> float | None:
    """Compute the p-th percentile (0-100) using linear interpolation
    between closest ranks (matches numpy's default 'linear' method)."""
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]

    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return sorted_vals[lower_idx]
    fraction = rank - lower_idx
    return sorted_vals[lower_idx] + (sorted_vals[upper_idx] - sorted_vals[lower_idx]) * fraction


def latency_summary(values: list[float]) -> dict:
    """Returns average/p50/p95/p99 for a list of real latency measurements.
    p99 is only included once there are enough samples (>=20) to be
    meaningful; otherwise it's reported as None rather than a noisy guess."""
    if not values:
        return {"count": 0, "avg": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "avg": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2) if len(values) >= 20 else None,
    }
