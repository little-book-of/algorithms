def dec_to_bin(n: int) -> str:
    """Convert a non-negative integer to binary string (no prefix)."""
    if n < 0:
        raise ValueError("Only non-negative integers supported.")
    if n == 0:
        return "0"

    bits = []
    while n > 0:
        bits.append(str(n % 2))
        n //= 2
    return "".join(reversed(bits))

def main():
    try:
        text = input("Enter a non-negative integer: ").strip()
        n = int(text)
        print(dec_to_bin(n))
    except ValueError:
        print("Invalid input.")

if __name__ == "__main__":
    main()
