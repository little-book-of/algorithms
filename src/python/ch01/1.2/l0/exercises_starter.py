def exercise_1():
    return 326 + 589

def exercise_2():
    return 704 - 259

def exercise_3():
    return 38 * 12

def exercise_4():
    q, r = divmod(123, 7)
    return q, r

def main():
    print("Exercise 1 (326+589):", exercise_1())
    print("Exercise 2 (704-259):", exercise_2())
    print("Exercise 3 (38*12):", exercise_3())
    q, r = exercise_4()
    print("Exercise 4 (123//7, 123%7):", q, r)


if __name__ == "__main__":
    main()
