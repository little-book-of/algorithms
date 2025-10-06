# ch01 / 1.4 / l0 — Overflow & Precision (Beginner, Lean 4)

Computers store numbers with a fixed number of **bits**. If a result exceeds that
maximum, it **wraps around** (like an odometer) — this is **integer overflow**.
For decimals, values like `0.1` cannot be represented exactly in binary, causing tiny
**precision errors** with floats.

## What you’ll learn
- **Integer overflow by simulation**: wrap at 2^bits, signed vs unsigned (two’s complement).
- **Floating-point surprises**: `0.1 + 0.2 != 0.3`, adding tiny to huge does nothing.
- **Safer habits**: avoid `==` for floats; use a small **epsilon** tolerance.

## Files
- `Main.lean` — quick, runnable tour.
- `IntOverflowSim.lean` — simulate fixed-width integers with modulo 2^bits.
- `FloatPrecision.lean` — float pitfalls + safe comparisons.
- `Exercises.lean` — simple practice tasks.

## Build & Run
```bash
lake build
lake exe main
