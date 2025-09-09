from parity import is_even
from divisibility import is_divisible
from remainder import div_identity, week_shift, last_digit_of_power

def exercise_1(n: int) -> str:
    return "even" if is_even(n) else "odd"

def exercise_2() -> bool:
    return is_divisible(91, 7)

def exercise_3() -> int:
    # remainder of 100 divided by 9
    return div_identity(100, 9)[1]

def exercise_4() -> int:
    # Saturday=5, add 10 days on a 0..6 week
    return week_shift(5, 10)

def exercise_5() -> int:
    # last digit of 2^15
    return last_digit_of_power(2, 15)

def main():
    print("Exercise 1 (42 parity):", exercise_1(42))
    print("Exercise 2 (91 divisible by 7?):", exercise_2())
    print("Exercise 3 (100 % 9):", exercise_3())
    print("Exercise 4 (Sat+10):", exercise_4())
    print("Exercise 5 (last digit of 2^15):", exercise_5())

if __name__ == "__main__":
    main()
