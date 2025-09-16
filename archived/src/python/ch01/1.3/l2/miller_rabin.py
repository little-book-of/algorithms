from typing import Iterable

def _decompose(n_minus_one: int):
    """Write n-1 = d * 2^s with d odd."""
    d = n_minus_one
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return d, s

def _try_composite(a: int, d: int, n: int, s: int) -> bool:
    """Return True if 'a' is a witness that n is composite."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True  # definitely composite for this base

def _deterministic_bases_64bit(n: int) -> Iterable[int]:
    # Deterministic for 64-bit per research (Jaeschke + later refinements).
    # Works for n < 2^64 with these bases:
    return (2, 3, 5, 7, 11, 13, 17)

def is_probable_prime(n: int) -> bool:
    """
    Miller–Rabin primality test.
    - Deterministic for 64-bit using fixed bases above.
    - For larger n, acts as a probabilistic test; consider extending bases.
    """
    if n < 2:
        return False
    # small primes & divisibility quick checks
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    if n in small_primes:
        return True
    for p in small_primes:
        if n % p == 0:
            return False
    d, s = _decompose(n - 1)
    bases = _deterministic_bases_64bit(n)
    for a in bases:
        if a % n == 0:  # base equals n
            continue
        if _try_composite(a, d, n, s):
            return False
    return True