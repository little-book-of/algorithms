def naive_mul(x: int, y: int) -> int:
    """Use Python's * as baseline (Python already uses optimized methods)."""
    return x * y

def main():
    a, b = 1234, 5678
    print("Naive multiplication:", naive_mul(a, b))
    print("Builtin multiplication:", a * b)

if __name__ == "__main__":
    main()
