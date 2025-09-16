from int_arith import (
    INT32_MAX, INT32_MIN,
    add_i32_checked, add_i32_wrapping, add_i32_saturating,
    add_i64_checked, add_i64_wrapping, add_i64_saturating,
)
from float_num import almost_equal, kahan_sum, pairwise_sum, machine_epsilon, ulp_diff
from fixed_point import parse_dollars_to_cents, format_cents, Ledger

def main():
    print("=== Integers: checked / wrapping / saturating ===")
    print("int32 max:", INT32_MAX)
    try:
        add_i32_checked(INT32_MAX, 1)
    except OverflowError as e:
        print("checked add (i32): overflow trapped ✔")

    print("wrapping add (i32):", add_i32_wrapping(INT32_MAX, 1))  # -> INT32_MIN
    print("saturating add (i32):", add_i32_saturating(INT32_MAX, 1))  # -> INT32_MAX

    print("wrapping add (i64):", add_i64_wrapping(9_223_372_036_854_775_607, 100))  # wraps
    print("saturating add (i64):", add_i64_saturating(9_223_372_036_854_775_700, 200))

    print("\n=== Floats: compare, sum, epsilon ===")
    x = 0.1 + 0.2
    print("0.1 + 0.2 =", repr(x))
    print("Direct equality with 0.3?", x == 0.3)
    print("almost_equal(x, 0.3):", almost_equal(x, 0.3, rel=1e-12, abs_=1e-12))
    arr = [1.0/(i+1) for i in range(100000)]
    print("naive vs pairwise vs kahan (first 1e5 harmonic terms):")
    # For speed in demo, just compare pairwise and kahan
    print("pairwise_sum:", pairwise_sum(arr))
    print("kahan_sum   :", kahan_sum(arr))
    print("machine epsilon:", machine_epsilon())
    print("ULP(1.0, nextafter(1.0, 2.0)) =", ulp_diff(1.0, float.fromhex('0x1.0000000000001p+0')))

    print("\n=== Fixed-point money (int cents) ===")
    a = Ledger(parse_dollars_to_cents("10.00"))
    b = Ledger(0)
    a.transfer_to(b, parse_dollars_to_cents("3.35"))
    print("A:", format_cents(a.balance_cents), "B:", format_cents(b.balance_cents))
    a.deposit(parse_dollars_to_cents("0.05"))
    print("A after 5 cents deposit:", format_cents(a.balance_cents))

if __name__ == "__main__":
    main()
