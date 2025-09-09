import math
from typing import Iterable, List

def almost_equal(x: float, y: float, rel: float = 1e-12, abs_: float = 1e-12) -> bool:
    """Robust float comparison: works near zero and for large magnitudes."""
    return abs(x - y) <= max(rel * max(abs(x), abs(y)), abs_)

def kahan_sum(xs: Iterable[float]) -> float:
    """Compensated summation (Kahan). Great for small-to-medium vectors."""
    s = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s

def pairwise_sum(xs: List[float]) -> float:
    """Pairwise (tree) summation. Stable and parallel-friendly."""
    arr = list(xs)
    n = len(arr)
    if n == 0:
        return 0.0
    while n > 1:
        nxt = []
        it = iter(arr[:n])
        for a in it:
            try:
                b = next(it)
            except StopIteration:
                nxt.append(a)
                break
            nxt.append(a + b)
        arr[:len(nxt)] = nxt
        n = len(nxt)
    return arr[0]

def ulp_diff(a: float, b: float) -> int:
    """ULP distance between two floats. Useful for reproducibility checks."""
    import struct
    ai = struct.unpack('!q', struct.pack('!d', a))[0]
    bi = struct.unpack('!q', struct.pack('!d', b))[0]
    # Handle negative ordering in two's complement ordering of doubles.
    if ai < 0: ai = 0x8000_0000_0000_0000 - ai
    if bi < 0: bi = 0x8000_0000_0000_0000 - bi
    return abs(ai - bi)

def machine_epsilon() -> float:
    """Smallest eps where 1.0 + eps != 1.0 for this platform."""
    eps = 1.0
    while 1.0 + eps / 2.0 != 1.0:
        eps /= 2.0
    return eps
