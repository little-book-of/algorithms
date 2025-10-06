from extgcd_inv import invmod
from totient import phi
from miller_rabin import is_probable_prime
from crt import crt

def exercise_1():
    # modular inverse 7 mod 13
    return invmod(7, 13)

def exercise_2():
    # phi(10) and verify Euler’s theorem for a=3
    val = phi(10)
    check = pow(3, val, 10) == 1
    return val, check

def exercise_3():
    # Fermat compositeness check on 341 (Carmichael)
    a = 2
    return pow(a, 340, 341)  # equals 1 despite composite

def exercise_4():
    # Use is_probable_prime on a few values
    return [n for n in [97, 341, 561, 569] if is_probable_prime(n)]

def exercise_5():
    # CRT small system: x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7)
    sol, mod = crt([(1, 4), (2, 5), (3, 7)])
    return sol, mod

def main():
    print("Exercise 1 invmod(7,13):", exercise_1())
    print("Exercise 2 phi(10) & check:", exercise_2())
    print("Exercise 3 Fermat 341 result:", exercise_3())
    print("Exercise 4 probable primes:", exercise_4())
    print("Exercise 5 CRT:", exercise_5())

if __name__ == "__main__":
    main()