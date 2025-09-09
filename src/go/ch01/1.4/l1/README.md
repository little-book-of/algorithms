# 1.4 L1 — Overflow & Precision in Real Systems (Go)

Practical patterns you can reuse across languages:
- **Integers:** checked / wrapping / saturating add (i32/i64) and checked mul.
- **Floats:** relative+absolute tolerance compare, Kahan & pairwise summation,
  machine epsilon probe, ULP distance.
- **Money:** fixed-point `int64` cents utilities and a tiny ledger.

## Files
- `demo_overflow_precision_practical.go` — runnable tour of the APIs.
- `int_arith.go` — i32/i64 checked, wrapping, saturating ops; checked mul.
- `float_num.go` — `AlmostEqual`, `KahanSum`, `PairwiseSum`, `MachineEpsilon`, `ULPDiff`.
- `fixed_point.go` — parse/format cents; minimal ledger with deposits/transfers.
- `exercises_starter.go` — tasks mirroring the text.
- `overflow_precision_practical_test.go` — `go test -v` sanity checks.

## Run
```bash
go run .
go test -v
```

## Notes
- Go’s fixed-width integers wrap by spec. “Checked” ops here prevent wrap by range testing before doing the operation. “Wrapping” ops make the intent explicit.
- For currency, prefer fixed-point or big decimals. Here we use int cents (fast, deterministic). Parsing truncates beyond 2 decimals (documented policy).