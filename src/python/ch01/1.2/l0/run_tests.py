def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {got}")

def test_ops():
    assert_eq(478 + 259, 737, "addition")
    assert_eq(503 - 78, 425, "subtraction")
    assert_eq(214 * 3, 642, "multiplication")
    q, r = divmod(47, 5)
    assert_eq(q, 9, "division quotient")
    assert_eq(r, 2, "division remainder")

def test_exercises():
    from exercises_starter import exercise_1, exercise_2, exercise_3, exercise_4
    assert_eq(exercise_1(), 915)
    assert_eq(exercise_2(), 445)
    assert_eq(exercise_3(), 456)
    q, r = exercise_4()
    assert_eq(q, 17)
    assert_eq(r, 4)

def main():
    test_ops()
    test_exercises()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
