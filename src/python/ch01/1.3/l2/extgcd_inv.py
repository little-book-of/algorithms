from typing import Tuple

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Return (g, x, y) such that a*x + b*y = g = gcd(a, b).
    Iterative version; x, y are Bézout coefficients.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    # old_r is gcd, and a*old_s + b*old_t = old_r
    return old_r, old_s, old_t

def invmod(a: int, m: int) -> int:
    """
    Return modular inverse x with (a*x) % m == 1, raising ValueError if not coprime.
    Result is in range [0, m-1].
    """
    if m <= 0:
        raise ValueError("modulus must be positive")
    g, x, _ = extended_gcd(a, m)
    if g != 1 and g != -1:
        raise ValueError("inverse does not exist; gcd(a, m) != 1")
    return x % m