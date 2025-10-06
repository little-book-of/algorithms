# ch01 / 1.4 / l0 — Overflow & Precision (Beginner, Erlang)

Computers store numbers with a fixed number of **bits**. When a result exceeds
that maximum, it **wraps around** (like an odometer) — this is **integer overflow**.
For decimals, values like `0.1` cannot be represented exactly in binary, causing tiny
**precision errors** with floats.

## What you’ll learn
- **Integer overflow by simulation**: 4/8/16/32-bit wraparound (unsigned & signed).
- **Floating-point surprises**: `0.1 + 0.2 != 0.3`, adding tiny to huge does nothing.
- **Safer habits**: avoid `==` for floats; use a small **epsilon** tolerance.

## Files
- `demo_overflow_precision_beginner.erl` — quick, runnable tour.
- `int_overflow_sim.erl` — simulate fixed-width integers in Erlang.
- `float_precision.erl` — float pitfalls + safe comparisons.
- `exercises_starter.erl` — simple practice tasks.
- `overflow_precision_beginner_tests.erl` — EUnit sanity checks.

## Run
```bash
rebar3 compile
erl -pa _build/default/lib/*/ebin -s demo_overflow_precision_beginner main -s init stop
erl -pa _build/default/lib/*/ebin -s exercises_starter main -s init stop
