from exercises_starter import exercise_1, exercise_2, exercise_3, exercise_4

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {actual}")

def main():
    b, o, h = exercise_1()
    assert_eq(int(b, 2), 100)
    assert_eq(int(o, 8), 100)
    assert_eq(int(h, 16), 100)

    assert_eq(exercise_2(), "11111001")
    assert_eq(exercise_3(), True)
    assert_eq(exercise_4(), -7)

    print("All tests passed ✔")

if __name__ == "__main__":
    main()
