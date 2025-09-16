# ch01 / 1.3 / l2 — Advanced Number Theory (Lean 4)

Algorithms included:
- Extended Euclid & modular inverse
- Euler’s totient φ(n) (trial factorization, didactic)
- Miller–Rabin primality test (deterministic base set for 64-bit)
- Chinese Remainder Theorem (pairwise coprime moduli)

## Files
- Main.lean — quick tour (invmod, φ, Miller–Rabin, CRT) + runs exercises.
- ExtGcdInv.lean — extendedGCD on Int, invMod for Int with Nat modulus.
- Totient.lean — phi(n : Nat) via trial factorization.
- MillerRabin.lean — isProbablePrime64 (Nat) with fixed bases.
- CRT.lean — crtPair / crt for pairwise coprime moduli.
- Exercises.lean — small tasks mirroring the text.

## Build & Run
lake build
lake exe main