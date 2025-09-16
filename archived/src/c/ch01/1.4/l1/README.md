# 1.4 L1 — Overflow & Precision in Real Systems (C)

Production-minded patterns:
- **Integers:** checked / wrapping / saturating add (i32/i64) and checked mul.
- **Floats:** relative+absolute tolerance compare, Kahan & pairwise summation,
  machine epsilon probe, ULP distance.
- **Money:** fixed-point `int64_t` cents utilities and a tiny ledger.

## Build & Run
```bash
make
./demo_overflow_precision_practical
./exercises
./tests.sh
```

## Notes
- Signed overflow is UB in C; all “checked” ops guard before performing the arithmetic. “Wrapping” ops use unsigned casts to force two’s-complement wrap.
- For currency, we keep balances in int cents (int64_t) for exactness.
- Consider adding sanitizers while developing: `make SAN="-fsanitize=undefined,address"`.