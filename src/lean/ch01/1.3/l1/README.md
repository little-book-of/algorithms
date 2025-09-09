# ch01 / 1.3 / l1 — Classical Number Theory Tools (Lean 4)

Euclid's GCD, LCM via GCD, modular arithmetic identities with normalization, and a fraction reducer with canonical sign.

## Files
- Main.lean — prints a quick tour (gcd/lcm, modular identities, fraction reduction) and runs exercises.
- GcdLcm.lean — Euclid's algorithm (Nat), lcm via gcd; simple Int wrappers where needed.
- ModularIdentities.lean — mod normalization and (a±b)%m, (a*b)%m utilities (Int inputs, Nat modulus).
- Fractions.lean — reduceFraction for Int numerators/denominators with denominator > 0.
- Exercises.lean — tasks mirroring the text.

## Build & Run
lake build
lake exe main