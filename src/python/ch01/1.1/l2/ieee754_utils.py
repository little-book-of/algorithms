import struct
from typing import Tuple

# IEEE-754 double precision (binary64) parameters
MANT_BITS = 52
EXP_BITS = 11
EXP_BIAS = 1023

def float_to_bits(x: float) -> int:
    """Return the 64-bit integer bit pattern of a Python float (IEEE-754 binary64)."""
    return struct.unpack('>Q', struct.pack('>d', x))[0]

def bits_to_float(bits: int) -> float:
    """Return the float whose IEEE-754 bit pattern is `bits` (64-bit)."""
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

def decompose(x: float) -> Tuple[int, int, int]:
    """
    Decompose float into (sign, exponent_raw, mantissa).
    - sign: 0 or 1
    - exponent_raw: 0..2047 (11 bits)
    - mantissa: 0..(2**52-1)
    """
    b = float_to_bits(x)
    sign = (b >> 63) & 0x1
    exponent_raw = (b >> MANT_BITS) & ((1 << EXP_BITS) - 1)
    mantissa = b & ((1 << MANT_BITS) - 1)
    return sign, exponent_raw, mantissa

def classify(x: float) -> str:
    """Return 'nan', 'inf', 'zero', 'subnormal', or 'normal' for x."""
    sign, e, m = decompose(x)
    if e == (1 << EXP_BITS) - 1:
        return 'nan' if m != 0 else 'inf'
    if e == 0:
        return 'zero' if m == 0 else 'subnormal'
    return 'normal'

def components_pretty(x: float) -> str:
    sign, e, m = decompose(x)
    kind = classify(x)
    return f"x={x!r} kind={kind} sign={sign} exponent_raw={e} mantissa=0x{m:013x}"

def build(sign: int, exponent_raw: int, mantissa: int) -> float:
    """Build a float from IEEE-754 fields (no validation beyond bit widths)."""
    sign = (sign & 1) << 63
    e = (exponent_raw & ((1 << EXP_BITS) - 1)) << MANT_BITS
    m = mantissa & ((1 << MANT_BITS) - 1)
    return bits_to_float(sign | e | m)

def exponent_unbiased(exponent_raw: int) -> int:
    """Convert raw exponent to unbiased exponent (for normal numbers)."""
    return exponent_raw - EXP_BIAS

def is_subnormal(x: float) -> bool:
    return classify(x) == 'subnormal'

def is_inf(x: float) -> bool:
    return classify(x) == 'inf'

def is_nan(x: float) -> bool:
    return classify(x) == 'nan'
