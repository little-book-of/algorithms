def gcd(a: int, b: int) -> int:
    """Euclid's algorithm (iterative). Returns non-negative gcd."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """Least common multiple using gcd. lcm(0, b) = 0 by convention."""
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)
