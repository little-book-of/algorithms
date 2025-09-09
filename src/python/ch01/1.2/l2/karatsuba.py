def karatsuba(x: int, y: int) -> int:
    """Recursive Karatsuba multiplication."""
    if x < 10 or y < 10:
        return x * y

    n = max(x.bit_length(), y.bit_length())
    m = n // 2

    high1, low1 = divmod(x, 1 << m)
    high2, low2 = divmod(y, 1 << m)

    z0 = karatsuba(low1, low2)
    z2 = karatsuba(high1, high2)
    z1 = karatsuba(low1 + high1, low2 + high2) - z0 - z2

    return (z2 << (2 * m)) + (z1 << m) + z0

def main():
    a, b = 1234, 5678
    print("Karatsuba result:", karatsuba(a, b))
    print("Expected:", a * b)

if __name__ == "__main__":
    main()
