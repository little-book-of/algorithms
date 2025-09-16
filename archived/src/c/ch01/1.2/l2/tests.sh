#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

echo "[test] demo_multiplication"
./demo_multiplication | grep -q "Naive multiplication:"

echo "[test] karatsuba_demo"
out=$(./karatsuba_demo)
echo "$out" | grep -q "Expected:"

echo "[test] modexp_demo"
out=$(./modexp_demo)
echo "$out" | grep -q "modexp(7,128,13) = 3"

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Karatsuba 31415926*27182818"
echo "$out" | grep -q "modexp 5^117 mod 19"
echo "$out" | grep -q "true"

echo "All tests passed ✔"
