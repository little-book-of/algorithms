#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling…"
make >/dev/null

echo "[test] demo_properties"
out=$(./demo_properties)
echo "$out" | grep -q "10 is_even? true"
echo "$out" | grep -q "7 is_odd? true"
echo "$out" | grep -q "12 divisible by 3? true"
echo "$out" | grep -q "14 divisible by 5? false"
echo "$out" | grep -q "17 = 5*3 + 2"
echo "$out" | grep -q "Saturday(5) + 10 days -> 1"
echo "$out" | grep -q "Last digit of 2^15:"
echo "$out" | grep -q "8"

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Exercise 1 (42 parity): even"
echo "$out" | grep -q "Exercise 2 (91 divisible by 7?): true"
echo "$out" | grep -q "Exercise 3 (100 % 9): 1"
echo "$out" | grep -q "Exercise 4 (Sat+10): 1"
echo "$out" | grep -q "Exercise 5 (last digit of 2^15): 8"

echo "All tests passed ✔"
