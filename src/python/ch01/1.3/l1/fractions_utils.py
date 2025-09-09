from gcd_lcm import gcd

def reduce_fraction(n: int, d: int) -> tuple[int, int]:
    """Return (num, den) in lowest terms. Raises on zero denominator."""
    if d == 0:
        raise ZeroDivisionError("denominator cannot be zero")
    g = gcd(n, d)
    num = n // g
    den = d // g
    # keep a canonical sign: denominator positive
    if den < 0:
        num, den = -num, -den
    return num, den
