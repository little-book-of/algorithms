def mod_add(a: int, b: int, m: int) -> int:
    """(a + b) % m computed safely via reductions."""
    return ((a % m) + (b % m)) % m

def mod_sub(a: int, b: int, m: int) -> int:
    """(a - b) % m with normalization to [0, m-1]."""
    return ((a % m) - (b % m)) % m

def mod_mul(a: int, b: int, m: int) -> int:
    """(a * b) % m via reductions (Python int is big, but keep the identity explicit)."""
    return ((a % m) * (b % m)) % m
