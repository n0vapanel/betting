"""Utility functions for numerical computations in the prediction engine."""

from __future__ import annotations

from typing import Iterable, List

def moving_average(data: Iterable[float], window: int) -> List[float]:
    """Compute the simple moving average of a sequence."""
    values = list(data)
    if window <= 0 or window > len(values):
        return []
    result: List[float] = []
    running_sum = sum(values[:window])
    result.append(running_sum / window)
    for i in range(window, len(values)):
        running_sum += values[i] - values[i - window]
        result.append(running_sum / window)
    return result

def scale_to_range(value: float, min_value: float, max_value: float) -> float:
    """Scale a value into the 0–1 range given its expected min and max."""
    if max_value <= min_value:
        return 0.0
    clamped = max(min(value, max_value), min_value)
    return (clamped - min_value) / (max_value - min_value)
