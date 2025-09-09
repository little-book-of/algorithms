def hashing_example(key: int, size: int) -> int:
    return key % size

def cyclic_day(start: int, shift: int) -> int:
    return (start + shift) % 7  # wrap around the week (0–6)

def modexp(a: int, b: int, m: int) -> int:
    """Square-and-multiply modular exponentiation."""
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
    print("Hashing: key=1234, size=10 ->", hashing_example(1234, 10))
    print("Cyclic: Saturday(5)+4 ->", cyclic_day(5, 4))  # 5=Saturday
    print("modexp(7, 128, 13) =", modexp(7, 128, 13))


if __name__ == "__main__":
    main()
