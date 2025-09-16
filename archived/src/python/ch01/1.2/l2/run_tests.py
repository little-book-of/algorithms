from karatsuba import karatsuba
from modexp import modexp

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {got}")

def test_karatsuba():
    a, b = 123456, 789012
    assert_eq(karatsuba(a, b), a * b, "karatsuba")

def test_modexp():
    assert_eq(modexp(7, 128, 13), pow(7, 128, 13), "modexp")
    assert_eq(modexp(5, 117, 19), pow(5, 117, 19), "modexp")

def test_exercises():
    from exercises_starter import exercise_1, exercise_2, exercise_3
    assert exercise_1() == 31415926 * 27182818
    assert exercise_2() == pow(5, 117, 19)
    assert exercise_3() is True

def main():
    test_karatsuba()
    test_modexp()
    test_exercises()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
