def modexp(a: int, b: int, m: int) -> int:
    """Modular exponentiation using square-and-multiply."""
    result = 1
    base = a % m
    exp = b
    while exp > 0:
        if exp & 1:
            result = (result * base) % m
        base = (base * base) % m
        exp >>= 1
    return result

def main():
    print("modexp(7,128,13) =", modexp(7, 128, 13))
    print("Python pow(7,128,13) =", pow(7, 128, 13))

if __name__ == "__main__":
    main()
