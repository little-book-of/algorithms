def exercise_1():
    return divmod(100, 9)

def exercise_2():
    return divmod(123, 11)

def exercise_3(n: int, d: int) -> bool:
    q, r = divmod(n, d)
    return n == d * q + r

def exercise_4(n: int) -> int:
    return n & 15  # n % 16

def exercise_5(n: int) -> bool:
    return n % 7 == 0

def main():
    print("Exercise 1 (100//9, 100%9):", exercise_1())
    print("Exercise 2 (123//11, 123%11):", exercise_2())
    print("Exercise 3 check (200, 23):", exercise_3(200, 23))
    print("Exercise 4 (37 % 16 via bitmask):", exercise_4(37))
    print("Exercise 5 (35 divisible by 7?):", exercise_5(35))


if __name__ == "__main__":
    main()
