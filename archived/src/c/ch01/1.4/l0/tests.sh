#!/usr/bin/env bash
set -euo pipefail

make >/dev/null

echo "[test] demo_overflow_precision_beginner"
out=$(./demo_overflow_precision_beginner)
echo "$out" | grep -q "255 \+ 1 (unsigned, wrap) -> 0"
echo "$out" | grep -q "127 \+ 1 (signed, wrap)   -> -128"
echo "$out" | grep -q "200 \+ 100 -> result=44 (unsigned), CF=true, OF=true"
echo "$out" | grep -q "Interpret result as signed: -212"
echo "$out" | grep -q "0.1 \+ 0.2 = 0.30000000000000004"
echo "$out" | grep -q "Direct equality with 0.3\? false"
echo "$out" | grep -q "Using epsilon: true"
echo "$out" | grep -q "sum = .* equal to 1.0\? false, almost_equal\? true"
echo "$out" | grep -q "1e16 \+ 1.0 = 10000000000000000"
echo "$out" | grep -q "sum_naive(\[0.1\]\*10) = "

echo "[test] exercises"
out2=$(./exercises)
echo "$out2" | grep -q "Ex1 (4-bit sequence from 14): 15 0 1"
echo "$out2" | grep -q "Ex2 (ten\*0.1): sum="
echo "$out2" | grep -q "eq1=false"
echo "$out2" | grep -q "almost=true"
echo "$out2" | grep -q "Ex3 (signed adds): -128 -2"
echo "$out2" | grep -q "Ex4 (u16 65535\+1): 0"

echo "All tests passed ✔"
