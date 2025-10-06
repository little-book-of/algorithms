#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling…"
make >/dev/null

echo "[test] demo_number_theory"
out=$(./demo_number_theory)
echo "$out" | grep -q "gcd(252, 198) = 18"
echo "$out" | grep -q "lcm(12, 18) = 36"
echo "$out" | grep -q "(x+y)%m vs mod_add:"
echo "$out" | grep -q "(x*y)%m vs mod_mul:"
echo "$out" | grep -q "reduce_fraction(84,126) = 2/3"

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Exercise 1 gcd(252,198): 18"
echo "$out" | grep -q "Exercise 2 lcm(12,18): 36"
echo "$out" | grep -q "Exercise 3 modular add check: "
echo "$out" | grep -q "Exercise 4 reduce 84/126: 2/3"
echo "$out" | grep -q "Exercise 5 smallest day multiple: 36"

echo "All tests passed ✔"