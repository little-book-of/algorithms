from gcd_lcm import gcd, lcm
from modular_identities import mod_add, mod_mul
from fractions_utils import reduce_fraction

def main():
    # GCD / LCM
    a, b = 252, 198
    print(f"gcd({a}, {b}) =", gcd(a, b))
    print(f"lcm(12, 18) =", lcm(12, 18))

    # Modular identities
    x, y, m = 123, 456, 7
    print(f"(x+y)%m vs mod_add:", (x + y) % m, mod_add(x, y, m))
    print(f"(x*y)%m vs mod_mul:", (x * y) % m, mod_mul(x, y, m))

    # Fraction reduction
    print("reduce_fraction(84, 126) =", reduce_fraction(84, 126))

if __name__ == "__main__":
    main()
