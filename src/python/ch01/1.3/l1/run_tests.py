from gcd_lcm import gcd, lcm
from modular_identities import mod_add, mod_sub, mod_mul
from fractions_utils import reduce_fraction
from exercises_starter import exercise_1, exercise_2, exercise_3, exercise_4, exercise_5

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} expected {expected}, got {got}")

def test_gcd_lcm():
    assert_eq(gcd(20, 14), 2)
    assert_eq(gcd(252, 198), 18)
    assert_eq(lcm(12, 18), 36)
    assert_eq(lcm(0, 5), 0)

def test_modular_identities():
    a, b, m = 123, 456, 7
    assert_eq(mod_add(a, b, m), (a + b) % m)
    assert_eq(mod_sub(a, b, m), (a - b) % m)
    assert_eq(mod_mul(a, b, m), (a * b) % m)
    assert_eq(mod_add(37, 85, 12), ((37 % 12) + (85 % 12)) % 12)

def test_fraction_reduce():
    assert_eq(reduce_fraction(84, 126), (2, 3))
    assert_eq(reduce_fraction(-6, 8), (-3, 4))
    assert_eq(reduce_fraction(6, -8), (-3, 4))
    assert_eq(reduce_fraction(-6, -8), (3, 4))

def test_exercises():
    assert_eq(exercise_1(), 18)
    assert_eq(exercise_2(), 36)
    lhs, rhs = exercise_3()
    assert_eq(lhs, rhs)
    assert_eq(exercise_4(), (2, 3))
    assert_eq(exercise_5(), 36)

def main():
    test_gcd_lcm()
    test_modular_identities()
    test_fraction_reduce()
    test_exercises()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
