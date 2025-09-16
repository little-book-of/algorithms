# ch01 / 1.4 / l0 — Overflow & Precision (Beginner, C)

Computers store numbers with a fixed number of **bits**. If a result goes past
the maximum value, it **wraps around** (like an odometer). For decimals, some
values (like `0.1`) cannot be represented exactly in binary, so tiny **precision
errors** appear.

## What you’ll learn
- **Integer overflow by simulation**: 8-bit and 32-bit wraparound (unsigned & signed).
- **Floating-point surprises**: `0.1 + 0.2 != 0.3`, losing small values next to huge ones.
- **Safer habits**: avoid `==` for floats; use a small tolerance (“epsilon”).

## Files
- `demo_overflow_precision_beginner.c` — quick tour you can run end-to-end.
- `int_overflow_sim.{h,c}` — helpers to simulate fixed-width integers in C.
- `float_precision.{h,c}` — small demos showing float pitfalls and safe comparison.
- `exercises_starter.c` — practice tasks with simple outputs.
- `tests.sh` — tiny sanity checks to verify behavior.

## Build & Run
```bash
make
./demo_overflow_precision_beginner
./exercises
./tests.sh
