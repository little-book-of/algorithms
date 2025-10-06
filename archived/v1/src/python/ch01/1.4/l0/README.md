# ch01 / 1.4 / l0 — Overflow & Precision (Beginner, Python)

Computers store numbers with a fixed number of **bits**. When a result goes past the
maximum that those bits can represent, it **wraps around** (like an odometer). For
decimals, some values (like `0.1`) cannot be represented exactly in binary, so tiny
**precision errors** appear.

## What you’ll learn
- **Integer overflow by simulation**: 8-bit and 32-bit wraparound (unsigned & signed).
- **Floating-point surprises**: `0.1 + 0.2 != 0.3`, losing small numbers next to huge ones.
- **Safer habits**: avoid `==` for floats; use a small tolerance (“epsilon”).

## Files
- `demo_overflow_precision_beginner.py` — quick tour you can run end-to-end.
- `int_overflow_sim.py` — helpers to simulate fixed-width integers in Python.
- `float_precision.py` — small demos showing float pitfalls and safe comparison.
- `exercises_starter.py` — practice tasks with TODOs.
- `run_tests.py` — tiny sanity tests to check your understanding.

## Run
```bash
python demo_overflow_precision_beginner.py
python exercises_starter.py
python run_tests.py
