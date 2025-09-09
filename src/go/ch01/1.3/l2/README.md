# ch01 / 1.3 / l2 — Advanced Number Theory (Go)

Core algorithms used in algorithmics & crypto:
- Extended Euclid & modular inverse
- Euler’s totient φ(n) (teaching version via trial factorization)
- Miller–Rabin primality test (deterministic for 64-bit)
- Chinese Remainder Theorem (pairwise coprime moduli)

## Files
- demo_number_theory_advanced.go — quick tour of invmod, φ, Miller–Rabin, CRT.
- extgcd_inv.go — Extended GCD and invmod for int64.
- totient.go — phi(n) via trial factorization (small/teaching).
- miller_rabin.go — 64-bit Miller–Rabin with fixed bases.
- crt.go — Chinese Remainder Theorem for coprime moduli.
- exercises_starter.go — practice tasks.
- number_theory_advanced_test.go — `go test` sanity checks.

## Run
go run .
go test -v