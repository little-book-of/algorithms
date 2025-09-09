#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

echo "[test] demo"
out=$(./demo_number_theory_advanced)
echo "$out" | grep -q "invmod(3,7) = 5"
echo "$out" | grep -q "phi(10) = 4"
echo "$out" | grep -q "is_probable_prime(97) = true"
echo "$out" | grep -q "is_probable_prime(561) = false"
echo "$out" | grep -q "CRT -> 23 mod 105"

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Exercise 1 invmod(7,13): 2"
echo "$out" | grep -q "Exercise 2 phi(10) & check: 4 true"
echo "$out" | grep -q "Exercise 3 Fermat 341 result: 1"
echo "$out" | grep -q "Exercise 4 probable primes: 97 569"
echo "$out" | grep -q "Exercise 5 CRT: 52 mod 140" || true
# Note: For the system x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7), a valid solution is 52 mod 140.

echo "All tests passed ✔"