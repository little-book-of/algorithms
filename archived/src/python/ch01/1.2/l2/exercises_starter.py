from karatsuba import karatsuba
from modexp import modexp

def exercise_1():
    return karatsuba(31415926, 27182818)

def exercise_2():
    return modexp(5, 117, 19)

def exercise_3():
    a, b = 12345, 67890
    return a * b == karatsuba(a, b)

def main():
    print("Exercise 1 (Karatsuba 31415926*27182818):", exercise_1())
    print("Exercise 2 (modexp 5^117 mod 19):", exercise_2())
    print("Exercise 3 (check Karatsuba correctness):", exercise_3())

if __name__ == "__main__":
    main()
