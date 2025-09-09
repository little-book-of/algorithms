# ch01 / 1.4 / l0 — Overflow & Precision (Beginner, Go)

Computers store numbers with a fixed number of **bits**. If a result goes past
the maximum those bits can represent, it **wraps around** (like an odometer).
For decimals, values like `0.1` cannot be represented exactly in binary, so tiny
**precision errors** appear.

## What you’ll learn
- **Integer overflow by simulation**: 8-bit and 32-bit wraparound (unsigned & signed).
- **Floating-point surprises**: `0.1 + 0.2 != 0.3`, losing small next to huge numbers.
- **Safer habits**: avoid `==` for floats; use a small tolerance (“epsilon”).

## Files
- `demo_overflow_precision_beginner.go` — quick tour you can run end-to-end.
- `int_overflow_sim.go` — helpers to simulate fixed-width integers in Go.
- `float_precision.go` — demos for float pitfalls and safe comparison.
- `exercises_starter.go` — practice tasks with simple outputs.
- `overflow_precision_beginner_test.go` — `go test` sanity checks.

## Run
```bash
go run .
go test -v
