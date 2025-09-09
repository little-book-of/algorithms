# ch01 / 1.3 / l1 — Classical Number Theory Tools (C)

Euclid's GCD, LCM via GCD, modular identities, and a fraction reducer.

## Files
- demo_number_theory.c — quick tour of gcd/lcm and modular identities.
- gcd_lcm.{h,c} — Euclid's algorithm (iterative) and lcm(a,b) using gcd.
- modular_identities.{h,c} — (a±b)%m and (a*b)%m helpers with normalization.
- fractions_utils.{h,c} — reduce_fraction(n,d) using gcd; canonical sign.
- exercises_starter.c — practice tasks.
- tests.sh — simple checks.

## Build & Run
make
./demo_number_theory
./exercises

## Test
./tests.sh