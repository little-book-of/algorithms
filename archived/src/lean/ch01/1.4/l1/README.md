# 1.4 L1 — Overflow & Precision in Real Systems (Lean 4)

Practical patterns you can port across languages:

- **Integers:** checked / wrapping / saturating add (i32/i64) and checked mul.
- **Floats:** relative+absolute tolerance compare, Kahan & pairwise summation,
  machine epsilon probe, ULP distance, nextAfterUp.
- **Money:** fixed-point `Int` cents utilities and a tiny ledger.

## Build & Run
```bash
lake build
lake exe main
```

## Notes
- Lean’s Int is arbitrary-precision; we simulate i32/i64 policies: checked (range guard), wrapping (two’s complement), and saturating (clamp).
- Float utilities follow the same policies used in other languages here: relative+absolute tolerances and numerically stable sums.
- Money is kept as int cents for exactness; parsing truncates >2 decimals by policy.
