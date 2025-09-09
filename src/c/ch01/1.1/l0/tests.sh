#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling..."
make >/dev/null

echo "[test] main"
./main | tee /tmp/out_main.txt
grep -q "Decimal: 42" /tmp/out_main.txt
grep -q "Binary :" /tmp/out_main.txt
grep -q "Binary string '1011' -> decimal: 11" /tmp/out_main.txt

echo "[test] exercises"
./exercises | tee /tmp/out_ex.txt
grep -q "Exercise 1 (19 -> binary): 10011" /tmp/out_ex.txt
grep -q "Exercise 2 ('10101' -> dec): 21" /tmp/out_ex.txt
grep -q "Exercise 3 (27 -> binary): 11011" /tmp/out_ex.txt
grep -q "== 63? true" /tmp/out_ex.txt

echo "All tests passed ✔"
