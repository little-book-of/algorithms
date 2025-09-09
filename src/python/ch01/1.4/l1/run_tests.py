from int_arith import (
    INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN,
    add_i32_checked, add_i32_wrapping, add_i32_saturating,
    add_i64_checked, add_i64_wrapping, add_i64_saturating,
    mul_i32_checked, mul_i64_checked
)
from float_num import almost_equal, kahan_sum, pairwise_sum, machine_epsilon, ulp_diff
from fixed_point import parse_dollars_to_cents, format_cents, Ledger

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} expected {expected}, got {got}")

def test_int32_add():
    try:
        add_i32_checked(INT32_MAX, 1)
        raise AssertionError("expected overflow")
    except OverflowError:
        pass
    assert_eq(add_i32_wrapping(INT32_MAX, 1), INT32_MIN)
    assert_eq(add_i32_saturating(INT32_MAX, 1), INT32_MAX)
    assert_eq(mul_i32_checked(46341, 46341), 2_147_488_281, "mul widen ok")
    try:
        mul_i32_checked(100_000, 100_000)
        raise AssertionError("expected mul overflow")
    except OverflowError:
        pass

def test_int64_add():
    try:
        add_i64_checked(INT64_MAX, 1)
        raise AssertionError("expected overflow")
    except OverflowError:
        pass
    assert_eq(add_i64_saturating(INT64_MAX, 1), INT64_MAX)
    assert add_i64_wrapping(INT64_MAX, 1) == INT64_MIN

def test_floats():
    x = 0.1 + 0.2
    assert not (x == 0.3)
    assert almost_equal(x, 0.3)
    xs = [1.0/(i+1) for i in range(50_000)]
    p = pairwise_sum(xs); k = kahan_sum(xs)
    # They should be close and better than naive (not computed here)
    assert abs(p - k) < 1e-10
    eps = machine_epsilon()
    assert (1.0 + eps) != 1.0
    assert ulp_diff(1.0, 1.0) == 0

def test_money():
    assert_eq(parse_dollars_to_cents("12.34"), 1234)
    assert_eq(parse_dollars_to_cents("-0.07"), -7)
    # Truncate extra decimals by policy
    assert_eq(parse_dollars_to_cents("1.239"), 123)
    a = Ledger(parse_dollars_to_cents("10.00"))
    b = Ledger(0)
    a.transfer_to(b, parse_dollars_to_cents("3.35"))
    assert_eq(format_cents(a.balance_cents), "6.65")
    assert_eq(format_cents(b.balance_cents), "3.35")

def main():
    test_int32_add()
    test_int64_add()
    test_floats()
    test_money()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
