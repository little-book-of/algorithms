def is_even(n: int) -> bool:
    """Return True if n is even (mod-based)."""
    return n % 2 == 0

def is_odd(n: int) -> bool:
    """Return True if n is odd."""
    return n % 2 != 0

def parity_bit(n: int) -> str:
    """Return 'even' or 'odd' using the last binary bit."""
    return "odd" if (n & 1) else "even"
