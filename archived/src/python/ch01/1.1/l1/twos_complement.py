def to_twos_complement(x: int, bits: int = 8) -> str:
    """Return the two's complement binary string of x with given bit width."""
    if x >= 0:
        return format(x, f"0{bits}b")
    else:
        return format((1 << bits) + x, f"0{bits}b")

def from_twos_complement(s: str) -> int:
    """Parse a two's complement binary string back into a signed integer."""
    bits = len(s)
    val = int(s, 2)
    if s[0] == "1":  # negative in two's complement
        val -= (1 << bits)
    return val

def main():
    print("+5:", to_twos_complement(5, 8))
    print("-5:", to_twos_complement(-5, 8))
    print("-1:", to_twos_complement(-1, 8))

    print("Parse back -5:", from_twos_complement("11111011"))

if __name__ == "__main__":
    main()
