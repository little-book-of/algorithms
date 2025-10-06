# ch01 / 1.3 / l2 — Advanced Number Theory (C)

Core algorithms:
- Extended Euclid & modular inverse
- Euler’s totient φ(n) (trial factorization, good for teaching/small n)
- Miller–Rabin primality test (deterministic for 64-bit bases)
- Chinese Remainder Theorem (pairwise coprime moduli)

## Files
- demo_number_theory_advanced.c — quick tour of invmod, φ, Miller–Rabin, CRT.
- extgcd_inv.{h,c} — extended_gcd and invmod for signed 64-bit.
- totient.{h,c} — phi(n) via trial factorization.
- miller_rabin.{h,c} — 64-bit Miller–Rabin with fixed bases.
- crt.{h,c} — Chinese remainder theorem for coprime moduli.
- exercises_starter.c — tasks mirroring the text.
- tests.sh — sanity checks (Carmichael numbers, known primes, CRT case).

## Build & Run
make
./demo_number_theory_advanced
./exercises

## Test
./tests.sh