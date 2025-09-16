from parity import is_even, is_odd, parity_bit
from divisibility import is_divisible, last_digit, divisible_by_10
from remainder import div_identity, week_shift, last_digit_of_power
from exercises_starter import exercise_1, exercise_2, exercise_3, exercise_4, exercise_5

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} expected {expected}, got {got}")

def test_parity():
    assert_eq(is_even(10), True)
    assert_eq(is_odd(7), True)
    assert_eq(parity_bit(11), "odd")
    assert_eq(parity_bit(8), "even")

def test_divisibility():
    assert_eq(is_divisible(12, 3), True)
    assert_eq(is_divisible(14, 5), False)
    assert_eq(last_digit(1234), 4)
    assert_eq(divisible_by_10(120), True)

def test_remainder():
    q, r = div_identity(17, 5)
    assert_eq((q, r), (3, 2))
    assert_eq(week_shift(5, 10), 1)  # Sat + 10 = Mon(1)
    assert_eq(last_digit_of_power(2, 15), 8)

def test_exercises():
    assert_eq(exercise_1(42), "even")
    assert_eq(exercise_2(), True)
    assert_eq(exercise_3(), 1)   # 100 % 9
    assert_eq(exercise_4(), 1)   # Sat+10 -> Mon
    assert_eq(exercise_5(), 8)   # last digit of 2^15

def main():
    test_parity()
    test_divisibility()
    test_remainder()
    test_exercises()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
