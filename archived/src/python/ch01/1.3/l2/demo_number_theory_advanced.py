from extgcd_inv import extended_gcd, invmod
from totient import phi
from miller_rabin import is_probable_prime
from crt import crt

def main():
    # Extended Euclid / inverse
    g, x, y = extended_gcd(240, 46)  # 240*x + 46*y = g
    print("extended_gcd(240,46) ->", (g, x, y))
    print("invmod(3, 7) =", invmod(3, 7))  # -> 5

    # Totient
    print("phi(10) =", phi(10))  # 4
    print("phi(36) =", phi(36))  # 12

    # Miller–Rabin (Carmichael vs primes)
    for n in [97, 561, 1105, 2_147_483_647]:  # prime, Carmichael, Carmichael, Mersenne prime
        print(f"is_probable_prime({n}) = {is_probable_prime(n)}")

    # CRT example: x≡2 (mod 3), x≡3 (mod 5), x≡2 (mod 7) -> 23 mod 105
    sol, mod = crt([(2, 3), (3, 5), (2, 7)])
    print("CRT ->", sol, "mod", mod)

if __name__ == "__main__":
    main()