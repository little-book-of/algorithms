import math
import sys
from ieee754_utils import (
    decompose, classify, float_to_bits, bits_to_float,
    exponent_unbiased, is_subnormal
)

def exercise_1_bits_of_one():
    """Return (sign, exponent_raw, mantissa) for 1.0 and unbiased exponent."""
    s, e, m = decompose(1.0)
    return s, e, m, exponent_unbiased(e)

def exercise_2_machine_epsilon_near_one():
    """Compute ULP(1.0) via nextafter and compare to sys.float_info.epsilon."""
    nxt = math.nextafter(1.0, math.inf)
    return nxt - 1.0, sys.float_info.epsilon

def exercise_3_classify_values():
    """Classify representative values."""
    values = [0.0, 1.0, float('inf'), float('nan'), 5e-324]
    return [(v, classify(v)) for v in values]

def exercise_4_roundtrip_bits(x: float):
    """Convert float to bits and back, return recovered value."""
    bits = float_to_bits(x)
    y = bits_to_float(bits)
    return bits, y

def exercise_5_find_first_subnormal():
    """Find the first subnormal above 0.0 using nextafter."""
    x = math.nextafter(0.0, 1.0)
    return x, is_subnormal(x)

def _demo():
    print("Ex1:", exercise_1_bits_of_one())  # expect sign=0, unbiased exponent=0, mantissa=0
    print("Ex2 ULP vs epsilon:", exercise_2_machine_epsilon_near_one())
    print("Ex3 classify:", exercise_3_classify_values())
    bits, y = exercise_4_roundtrip_bits(3.5)
    print(f"Ex4 roundtrip 3.5: bits=0x{bits:016x}, back={y}")
    print("Ex5 first subnormal > 0.0:", exercise_5_find_first_subnormal())

if __name__ == "__main__":
    _demo()
