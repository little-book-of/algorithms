#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

echo "[test] demo"
out=$(./demo_overflow_precision_practical)
echo "$out" | grep -q "checked add (i32): overflow trapped ✔"
echo "$out" | grep -q "wrapping add (i32): -2147483648"
echo "$out" | grep -q "saturating add (i32): 2147483647"
echo "$out" | grep -q "0.1 \+ 0.2 = 0.30000000000000004"
echo "$out" | grep -q "Direct equality with 0.3\? false"
echo "$out" | grep -q "almost_equal(x, 0.3): true"
echo "$out" | grep -q "machine epsilon:"
echo "$out" | grep -q "ULP(1.0, nextafter(1.0, 2.0)) = 1"
echo "$out" | grep -q "A: 6.65 B: 3.35"
echo "$out" | grep -q "A after 5 cents deposit: 6.70"

echo "[test] exercises"
ex=$(./exercises)
echo "$ex" | grep -q "Exercise 1 (policy): wrap=-2147483648 checked=OverflowError sat=2147483647"
echo "$ex" | grep -q "Exercise 2 (float compare): .* eq=false almost=true"
echo "$ex" | grep -q "Exercise 3 (pairwise vs kahan):"
echo "$ex" | grep -q "Exercise 4 (ledger): 12.00 0.34"

echo "All tests passed ✔"
