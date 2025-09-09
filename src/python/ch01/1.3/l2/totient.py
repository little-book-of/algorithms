def phi(n: int) -> int:
    """
    Euler’s totient φ(n): count of 1<=k<=n that are coprime to n.
    Teaching version: trial division factorization, O(sqrt n).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1 if p == 2 else 2  # after 2, check odds only
    if x > 1:  # leftover prime
        result -= result // x
    return result