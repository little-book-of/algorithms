def main():
    n, d = 47, 5
    q, r = divmod(n, d)
    print(f"{n} = {d}*{q} + {r}")

    n, d = 23, 7
    q, r = divmod(n, d)
    print(f"{n} = {d}*{q} + {r}")

    n, d = 100, 9
    q, r = divmod(n, d)
    print(f"{n} = {d}*{q} + {r}")


if __name__ == "__main__":
    main()
