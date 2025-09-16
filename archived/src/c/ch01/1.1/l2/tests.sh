#!/usr/bin/env bash
set -euo pipefail

echo "[build] compiling…"
make >/dev/null

echo "[test] demo"
./demo | tee /tmp/out_demo.txt
grep -q "Equal to 0.3? false" /tmp/out_demo.txt
grep -q "DBL_EPSILON:" /tmp/out_demo.txt
grep -q "ULP\(1.0\):" /tmp/out_demo.txt
grep -q "First subnormal > 0:" /tmp/out_demo.txt
grep -q "kind=normal" /tmp/out_demo.txt
grep -q "kind=zero" /tmp/out_demo.txt
grep -q "kind=inf" /tmp/out_demo.txt
grep -q "kind=nan" /tmp/out_demo.txt

echo "[test] exercises"
./exercises | tee /tmp/out_ex.txt
grep -q "Ex1: sign=0" /tmp/out_ex.txt
grep -q "Ex2: ULP(1.0)=" /tmp/out_ex.txt
grep -q "Ex3: zero -> zero" /tmp/out_ex.txt
grep -q "Ex4: 3.5 bits=0x" /tmp/out_ex.txt
grep -q "Ex5: built +inf=inf, built nan==nan? false" /tmp/out_ex.txt

echo "All tests passed ✔"
