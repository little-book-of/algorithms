def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {got}")

def test_division():
    assert_eq(divmod(47, 5), (9, 2))
    assert_eq(divmod(23, 7), (3, 2))
    assert_eq(divmod(100, 9), (11, 1))

def test_exercises():
    from exercises_starter import exercise_1, exercise_2, exercise_3, exercise_4, exercise_5
    assert_eq(exercise_1(), (11, 1))
    assert_eq(exercise_2(), (11, 2))
    assert_eq(exercise_3(200, 23), True)
    assert_eq(exercise_4(37), 5)  # 37 % 16 = 5
    assert_eq(exercise_5(35), True)

def main():
    test_division()
    test_exercises()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
