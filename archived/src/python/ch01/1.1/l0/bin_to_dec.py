def bin_to_dec(s: str) -> int:
    """Convert binary string -> decimal integer."""
    s = s.strip()
    if not s or any(ch not in "01" for ch in s):
        raise ValueError("Input must be a string of '0' and '1'.")

    value = 0
    for ch in s:
        value = value * 2 + (1 if ch == "1" else 0)
    return value

def main():
    try:
        s = input("Enter a binary string: ")
        print(bin_to_dec(s))
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
