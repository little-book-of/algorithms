from int_arith import (
    INT32_MAX, INT32_MIN,
    add_i32_checked, add_i32_wrapping, add_i32_saturating,
)
from float_num import almost_equal, kahan_sum, pairwise_sum, machine_epsilon
from fixed_point import parse_dollars_to_cents, format_cents, Ledger

def exercise_1_policy():
    """
    Choose addition policy for 3 counters:
    - packet_count (wrapping ok)
    - active_users (checked)
    - storage_bytes (saturating)
    Return results of (INT32_MAX + 1) under each policy.
    """
    wrap = add_i32_wrapping(INT32_MAX, 1)
    try:
        chk = add_i32_checked(INT32_MAX, 1)
    except OverflowError:
        chk = "OverflowError"
    sat = add_i32_saturating(INT32_MAX, 1)
    return wrap, chk, sat

def exercise_2_compare_floats():
    x = 0.1 + 0.2
    return x, (x == 0.3), almost_equal(x, 0.3, rel=1e-12, abs_=1e-12)

def exercise_3_summation():
    xs = [1.0/(i+1) for i in range(100000)]
    return pairwise_sum(xs), kahan_sum(xs)

def exercise_4_ledger():
    a = Ledger(parse_dollars_to_cents("12.34"))
    b = Ledger(0)
    a.transfer_to(b, parse_dollars_to_cents("0.34"))
    return format_cents(a.balance_cents), format_cents(b.balance_cents)

def main():
    print("Exercise 1 (policy):", exercise_1_policy())
    print("Exercise 2 (float compare):", exercise_2_compare_floats())
    p, k = exercise_3_summation()
    print("Exercise 3 (pairwise vs kahan):", p, k)
    print("Exercise 4 (ledger):", exercise_4_ledger())

if __name__ == "__main__":
    main()
