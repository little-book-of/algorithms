from twos_complement import to_twos_complement, from_twos_complement

def exercise_1():
    # Convert decimal 100 to binary, octal, hex
    n = 100
    return bin(n), oct(n), hex(n)

def exercise_2():
    # Write -7 in 8-bit two's complement
    return to_twos_complement(-7, 8)

def exercise_3():
    # Verify 0xFF == 255
    return (0xFF == 255)

def exercise_4():
    # Parse "11111001" as 8-bit two's complement
    return from_twos_complement("11111001")

def _demo():
    print("Exercise 1 (100):", exercise_1())        # ('0b1100100', '0o144', '0x64')
    print("Exercise 2 (-7):", exercise_2())        # '11111001'
    print("Exercise 3 (0xFF):", exercise_3())      # True
    print("Exercise 4 (11111001):", exercise_4())  # -7

if __name__ == "__main__":
    _demo()
