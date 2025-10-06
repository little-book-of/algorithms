def add_binary(x: int, y: int) -> str:
    """Add two integers and return result as binary string."""
    return bin(x + y)

def sub_binary(x: int, y: int) -> str:
    """Subtract and return result as binary string."""
    return bin(x - y)


def main():
    x, y = 0b1011, 0b0110  # 11 and 6
    print("x =", bin(x), "y =", bin(y))
    print("x + y =", add_binary(x, y))
    print("x - y =", sub_binary(x, y))


if __name__ == "__main__":
    main()
