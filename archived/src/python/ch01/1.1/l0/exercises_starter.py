from dec_to_bin import dec_to_bin
from bin_to_dec import bin_to_dec

def exercise_1_decimal_to_binary(n: int) -> str:
    return dec_to_bin(n)

def exercise_2_binary_to_decimal(s: str) -> int:
    return bin_to_dec(s)

def exercise_3_convert_27() -> str:
    return dec_to_bin(27)

def exercise_4_verify_literal() -> bool:
    return 0b111111 == 63

def exercise_5_why_binary() -> str:
    return "Electronics are reliable with two states (0/1), making binary simpler than decimal."

def _demo():
    print("Exercise 1 (19 -> binary):", exercise_1_decimal_to_binary(19))
    print("Exercise 2 ('10101' -> dec):", exercise_2_binary_to_decimal("10101"))
    print("Exercise 3 (27 -> binary):", exercise_3_convert_27())
    print("Exercise 4 (0b111111 == 63?):", exercise_4_verify_literal())
    print("Exercise 5 (why binary):", exercise_5_why_binary())

if __name__ == "__main__":
    _demo()
