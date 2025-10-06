from parity import is_even, is_odd, parity_bit
from divisibility import is_divisible
from remainder import div_identity, week_shift

def main():
    print("Parity:")
    for n in [10, 7]:
        print(f"  {n} is_even? {is_even(n)}  is_odd? {is_odd(n)}  (bit) {parity_bit(n)}")

    print("\nDivisibility:")
    print("  12 divisible by 3?", is_divisible(12, 3))
    print("  14 divisible by 5?", is_divisible(14, 5))

    print("\nRemainders & identity:")
    q, r = div_identity(17, 5)
    print(f"  17 = 5*{q} + {r}")

    print("\nClock (7-day week):")
    # 0=Mon, 5=Sat
    print("  Saturday(5) + 10 days ->", week_shift(5, 10))

if __name__ == "__main__":
    main()
