from typing import Iterable

def almost_equal(x: float, y: float, eps: float = 1e-9) -> bool:
    """Return True if x and y are within eps."""
    return abs(x - y) < eps

def repeat_add(value: float, times: int) -> float:
    """Add 'value' to 0.0 'times' times; shows tiny drift for decimals like 0.1."""
    s = 0.0
    for _ in range(times):
        s += value
    return s

def mix_large_small(large: float, small: float) -> float:
    """
    Adding a tiny number to a huge one may have no effect:
    e.g., 1e16 + 1 == 1e16 due to limited precision.
    """
    return large + small

def sum_list(xs: Iterable[float]) -> float:
    """Naive left-to-right sum (for beginners)."""
    s = 0.0
    for v in xs:
        s += v
    return s
