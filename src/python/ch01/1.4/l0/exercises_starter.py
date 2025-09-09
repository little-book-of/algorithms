from int_overflow_sim import add_unsigned, add_signed
from float_precision import almost_equal, repeat_add

# Exercise 1: 4-bit counter sequence starting at 14, add 1 three times.
def exercise_1():
    seq = []
    x = 14
    for _ in range(3):
        x = add_unsigned(x, 1, bits=4)
        seq.append(x)
    return seq  # expected: [15, 0, 1]

# Exercise 2: Does 0.1 added ten times equal exactly 1.0?
def exercise_2():
    s = repeat_add(0.1, 10)
    return s, (s == 1.0), almost_equal(s, 1.0)

# Exercise 3: Implement an 8-bit signed addition demo for 127 + 1 and -1 + (-1)
def exercise_3():
    a = add_signed(127, 1, bits=8)   # expect -128
    b = add_signed(-1, -1, bits=8)   # expect -2
    return a, b

# Exercise 4: Write a small helper that simulates unsigned 16-bit wrap for 65535 + 1
def exercise_4():
    return add_unsigned(65535, 1, bits=16)   # expect 0

def main():
    print("Ex1 (4-bit sequence):", exercise_1())
    print("Ex2 (ten 0.1):", exercise_2())
    print("Ex3 (signed adds):", exercise_3())
    print("Ex4 (u16 wrap):", exercise_4())

if __name__ == "__main__":
    main()
