def div_identity(n: int, d: int) -> tuple[int, int]:
    """
    Return (q, r) such that n = d*q + r and 0 <= r < |d|.
    Mirrors Python's // and % behavior for integers.
    """
    if d == 0:
        raise ZeroDivisionError("division by zero")
    q, r = divmod(n, d)
    return q, r

def week_shift(start: int, shift: int) -> int:
    """
    Wrap days of week with 0..6 (0=Mon). Return the day index after shift days.
    """
    return (start + shift) % 7

def last_digit_of_power(base: int, exp: int) -> int:
    """
    Compute the last decimal digit of base**exp using modular arithmetic.
    Uses Python's pow with modulus for clarity.
    """
    return pow(base, exp, 10)
