from gcd_lcm import gcd, lcm
from modular_identities import mod_add, mod_mul
from fractions_utils import reduce_fraction

def exercise_1():
    # gcd(252, 198)
    return gcd(252, 198)

def exercise_2():
    # lcm(12, 18)
    return lcm(12, 18)

def exercise_3():
    # (37+85)%12 equals ((37%12)+(85%12))%12
    lhs = (37 + 85) % 12
    rhs = mod_add(37, 85, 12)
    return lhs, rhs

def exercise_4():
    # reduce 84/126
    return reduce_fraction(84, 126)

def exercise_5():
    # smallest day multiple of 12 and 18 (i.e., LCM)
    return lcm(12, 18)

def main():
    print("Exercise 1 gcd(252,198):", exercise_1())
    print("Exercise 2 lcm(12,18):", exercise_2())
    print("Exercise 3 modular add check:", exercise_3())
    print("Exercise 4 reduce 84/126:", exercise_4())
    print("Exercise 5 smallest day multiple:", exercise_5())

if __name__ == "__main__":
    main()
