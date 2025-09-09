from typing import Iterable, Tuple, Optional
from extgcd_inv import invmod

def crt_pair(a1: int, m1: int, a2: int, m2: int) -> Tuple[int, int]:
    """
    Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2) for coprime m1,m2.
    Return (x, m1*m2) with x in [0, m1*m2-1].
    Raise ValueError if moduli not coprime or inconsistent.
    """
    from math import gcd
    if gcd(m1, m2) != 1:
        raise ValueError("moduli must be coprime for crt_pair")
    k = ((a2 - a1) % m2) * invmod(m1 % m2, m2) % m2
    x = (a1 + k * m1) % (m1 * m2)
    return x, m1 * m2

def crt(congruences: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Fold CRT across many congruences with pairwise coprime moduli.
    Returns (x, M) s.t. x ≡ ai (mod mi) for all (ai, mi).
    """
    it = iter(congruences)
    try:
        a, m = next(it)
    except StopIteration:
        raise ValueError("no congruences")
    for a2, m2 in it:
        a, m = crt_pair(a, m, a2, m2)
    return a, m