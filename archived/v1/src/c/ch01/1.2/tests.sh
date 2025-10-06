#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

echo "[test] demo_divmod"
out=$(./demo_divmod)
echo "$out" | grep -q "47 = 5*9 + 2"
echo "$out" | grep -q "23 = 7*3 + 2"
echo "$out" | grep -q "100 = 9*11 + 1"

echo "[test] modulo_examples"
out=$(./modulo_examples)
echo "$out" | grep -q "Hashing: key=1234, size=10 -> 4"
echo "$out" | grep -q "Cyclic: Saturday(5)+4 -> 2"
echo "$out" | grep -q "modexp(7,128,13) = 3"

echo "[test] efficiency_demo"
out=$(./efficiency_demo)
echo "$out" | grep -q "5 % 8 = 5, bitmask = 5"
echo "$out" | grep -q "12 % 8 = 4, bitmask = 4"
echo "$out" | grep -q "20 % 8 = 4, bitmask = 4"

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Exercise 1 (100//9,100%9): 11 1"
echo "$out" | grep -q "Exercise 2 (123//11,123%11): 11 2"
echo "$out" | grep -q "Exercise 3 check (200,23): true"
echo "$out" | grep -q "Exercise 4 (n % 16 via bitmask): 37 -> 5"
echo "$out" | grep -q "Exercise 5 (35 divisible by 7?): true"

echo "All tests passed ✔"
