#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling…"
make >/dev/null

echo "[test] demo"
./demo | tee /tmp/out_demo.txt
grep -q "Decimal: 42" /tmp/out_demo.txt
grep -q "Octal  : 52" /tmp/out_demo.txt
grep -q "Hex    : 2A" /tmp/out_demo.txt
grep -q "Octal literal 052 -> 42" /tmp/out_demo.txt
grep -q "Hex literal 0x2A -> 42" /tmp/out_demo.txt

echo "[test] exercises"
./exercises | tee /tmp/out_ex.txt
grep -q "Exercise 1 (100): oct=144 hex=64" /tmp/out_ex.txt
grep -q "Exercise 2 (-7 in 8-bit): 11111001" /tmp/out_ex.txt
grep -q "Exercise 3 (0xFF == 255?): true" /tmp/out_ex.txt
grep -q "Exercise 4 (parse '11111001'): -7" /tmp/out_ex.txt

echo "All tests passed ✔"
