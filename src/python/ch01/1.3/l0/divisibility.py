def is_divisible(a: int, b: int) -> bool:
    """True iff b divides a (b != 0)."""
    if b == 0:
        raise ZeroDivisionError("divisibility by zero undefined")
    return a % b == 0

def last_digit(n: int) -> int:
    """Return the last decimal digit of n."""
    return abs(n) % 10

def divisible_by_10(n: int) -> bool:
    """Quick rule: last digit is 0."""
    return last_digit(n) == 0
