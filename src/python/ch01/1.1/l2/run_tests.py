import math
import sys
from ieee754_utils import (
    decompose, classify, float_to_bits, bits_to_float,
    exponent_unbiased
)
from decimal import Decimal, getcontext

def assert_eq(x, y, msg=""):
    if x != y:
        raise AssertionError(f"{msg} Expected {y!r}, got {x!r}")

def assert_almost(x, y, eps=1e-18, msg=""):
    if abs(x - y) > eps:
        raise AssertionError(f"{msg} Expected ~{y!r}, got {x!r}")

def test_rounding_surprise():
    a = 0.1 + 0.2
    assert (a == 0.3) is False

def test_bits_one():
    s, e, m = decompose(1.0)
    assert_eq(s, 0, "sign(1.0)")
    # For 1.0, exponent_raw must be bias=1023, mantissa 0
    assert_eq(e, 1023, "exponent_raw(1.0)")
    assert_eq(m, 0, "mantissa(1.0)")
    assert_eq(exponent_unbiased(e), 0, "unbiased exponent of 1.0")

def test_roundtrip():
    x = 3.5
    bits = float_to_bits(x)
    y = bits_to_float(bits)
    assert_eq(y, x, "roundtrip 3.5")

def test_decimal_exact():
    getcontext().prec = 28
    a = Decimal('0.1') + Decimal('0.2')
    assert_eq(a == Decimal('0.3'), True, "decimal exact 0.1+0.2==0.3")

def test_ulps():
    # ULP near 1.0 equals sys.float_info.epsilon for IEEE-754 doubles
    nxt = math.nextafter(1.0, math.inf)
    ulp = nxt - 1.0
    assert_almost(ulp, sys.float_info.epsilon, 0, "ULP near 1.0")

def test_classify():
    assert_eq(classify(0.0), 'zero')
    assert_eq(classify(1.0), 'normal')
    assert_eq(classify(float('inf')), 'inf')
    assert_eq(classify(float('-inf')), 'inf')
    assert_eq(classify(5e-324), 'subnormal')

def main():
    test_rounding_surprise()
    test_bits_one()
    test_roundtrip()
    test_decimal_exact()
    test_ulps()
    test_classify()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
