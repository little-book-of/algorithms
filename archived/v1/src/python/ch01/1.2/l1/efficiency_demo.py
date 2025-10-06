def main():
    for n in [5, 12, 20]:
        print(f"{n} % 8 = {n % 8}, bitmask = {n & 7}")


if __name__ == "__main__":
    main()
