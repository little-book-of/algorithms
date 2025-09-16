def main():
    a, b = 478, 259
    print("Addition:", a, "+", b, "=", a + b)

    a, b = 503, 78
    print("Subtraction:", a, "-", b, "=", a - b)

    a, b = 214, 3
    print("Multiplication:", a, "*", b, "=", a * b)

    n, d = 47, 5
    q, r = divmod(n, d)
    print(f"Division: {n} / {d} -> quotient {q}, remainder {r}")


if __name__ == "__main__":
    main()
