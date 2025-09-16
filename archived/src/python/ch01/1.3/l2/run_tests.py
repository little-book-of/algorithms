from math import gcd
from extgcd_inv import extended_gcd, invmod
from totient import phi
from miller_rabin import is_probable_prime
from crt import crt

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} expected {expected}, got {got}")

def test_extended_gcd_inv():
    g,x,y = extended_gcd(240, 46)
    assert_eq(g, gcd(240,46))
    assert 240*x + 46*y == g
    assert_eq(invmod(3,7), 5)
    try:
        invmod(6, 9)
        raise AssertionError("invmod should fail for non-coprime")
    except ValueError:
        pass

def test_phi():
    assert_eq(phi(1), 1)
    assert_eq(phi(10), 4)
    assert_eq(phi(36), 12)
    assert_eq(phi(97), 96)  # prime

def test_miller_rabin():
    # True primes
    for p in [2,3,5,7,11,97,569,2_147_483_647]:
        assert is_probable_prime(p)
    # Obvious composites
    for c in [1,4,6,8,9,10,12,100]:
        assert not is_probable_prime(c)
    # Carmichael numbers: Fermat would be fooled; MR should reject
    for c in [561, 1105, 1729, 2465]:
        assert not is_probable_prime(c)

def test_crt():
    x, m = crt([(2,3),(3,5),(2,7)])
    assert_eq((x % 3, x % 5, x % 7), (2,3,2))
    assert_eq(m, 105)

def main():
    test_extended_gcd_inv()
    test_phi()
    test_miller_rabin()
    test_crt()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()