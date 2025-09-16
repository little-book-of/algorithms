#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling…"
make >/dev/null

echo "[test] demo_basic_ops"
out=$(./demo_basic_ops)
echo "$out" | grep -q "Addition: 478 + 259 = 737"
echo "$out" | grep -q "Subtraction: 503 - 78 = 425"
echo "$out" | grep -q "Multiplication: 214 * 3 = 642"
echo "$out" | grep -q "Division: 47 / 5 -> quotient 9, remainder 2"

echo "[test] binary_ops"
out=$(./binary_ops)
echo "$out" | grep -q "x + y ="
echo "$out" | grep -q "x - y ="

echo "[test] exercises"
out=$(./exercises)
echo "$out" | grep -q "Exercise 1 (326+589): 915"
echo "$out" | grep -q "Exercise 2 (704-259): 445"
echo "$out" | grep -q "Exercise 3 (38*12): 456"
echo "$out" | grep -q "Exercise 4 (123//7, 123%7): 17 4"

echo "All tests passed ✔"
