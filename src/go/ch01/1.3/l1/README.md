# ch01 / 1.3 / l1 — Classical Number Theory Tools (Go)

Euclid's GCD, LCM via GCD, modular arithmetic identities, and a fraction reducer.

## Files
- demo_number_theory.go — quick tour of gcd/lcm and modular identities.
- gcd_lcm.go — Euclid (iterative) and lcm(a,b) via gcd.
- modular_identities.go — (a±b)%m and (a*b)%m helpers (normalized).
- fractions_utils.go — reduceFraction(n,d) with canonical sign.
- exercises_starter.go — tasks mirroring the text.
- number_theory_test.go — `go test` sanity checks.

## Run
go run .
go test -v