# ch01 / 1.3 / l2 — Advanced Number Theory (Python)

Core algorithms used in modern algorithmics & crypto:
- Extended Euclid & modular inverse
- Euler’s totient φ(n)
- Miller–Rabin primality test (deterministic for 64-bit)
- Tiny CRT helper

## Files
- demo_number_theory_advanced.py — quick tour of invmod, φ, Miller–Rabin, CRT.
- extgcd_inv.py — extended_gcd(a,b) and invmod(a,m).
- totient.py — phi(n) via trial factorization (good for teaching/small n).
- miller_rabin.py — is_probable_prime(n) with safe bases for 64-bit.
- crt.py — Chinese Remainder Theorem for coprime moduli.
- exercises_starter.py — tasks mirroring the text.
- run_tests.py — sanity tests (Carmichael numbers, known primes, etc.).

## Run
python demo_number_theory_advanced.py
python exercises_starter.py
python run_tests.py