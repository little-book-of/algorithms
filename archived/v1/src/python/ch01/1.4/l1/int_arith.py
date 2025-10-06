from typing import Tuple

INT32_MIN, INT32_MAX = -2_147_483_648, 2_147_483_647
INT64_MIN, INT64_MAX = -9_223_372_036_854_775_808, 9_223_372_036_854_775_807

# ---------- helpers to convert to signed from masked (two's complement) ----------
def _to_signed(masked: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    full = (1 << bits)
    return masked - full if (masked & sign) else masked

# ---------- i32 ops ----------
def add_i32_checked(a: int, b: int) -> int:
    s = a + b
    if s < INT32_MIN or s > INT32_MAX:
        raise OverflowError("int32 overflow")
    return s

def add_i32_wrapping(a: int, b: int) -> int:
    s = (a + b) & 0xFFFFFFFF
    return _to_signed(s, 32)

def add_i32_saturating(a: int, b: int) -> int:
    s = a + b
    if s > INT32_MAX: return INT32_MAX
    if s < INT32_MIN: return INT32_MIN
    return s

def mul_i32_checked(a: int, b: int) -> int:
    # widen in Python, then bounds-check for i32 range
    p = a * b
    if p < INT32_MIN or p > INT32_MAX:
        raise OverflowError("int32 overflow (mul)")
    return p

# ---------- i64 ops ----------
def add_i64_checked(a: int, b: int) -> int:
    s = a + b
    if s < INT64_MIN or s > INT64_MAX:
        raise OverflowError("int64 overflow")
    return s

def add_i64_wrapping(a: int, b: int) -> int:
    s = (a + b) & 0xFFFFFFFFFFFFFFFF
    return _to_signed(s, 64)

def add_i64_saturating(a: int, b: int) -> int:
    s = a + b
    if s > INT64_MAX: return INT64_MAX
    if s < INT64_MIN: return INT64_MIN
    return s

def mul_i64_checked(a: int, b: int) -> int:
    p = a * b
    if p < INT64_MIN or p > INT64_MAX:
        raise OverflowError("int64 overflow (mul)")
    return p
