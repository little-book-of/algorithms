# 1.4 L1 — Overflow & Precision in Real Systems (Python)

Practical patterns you can port across languages:
- **Integers:** checked / wrapping / saturating add (i32/i64), plus simple checked mul.
- **Floats:** relative+absolute epsilon compare, Kahan & pairwise summation, ULP probes.
- **Money:** fixed-point `int64` cents utilities and a tiny in-memory ledger.

## Files
- `demo_overflow_precision_practical.py` — runnable tour of the APIs below.
- `int_arith.py` — int32/int64 checked, wrapping, saturating ops; checked mul.
- `float_num.py` — `almost_equal`, `kahan_sum`, `pairwise_sum`, `ulp_diff`.
- `fixed_point.py` — parse/format cents; a minimal ledger with deposits/transfers.
- `exercises_starter.py` — tasks mirroring the text.
- `run_tests.py` — sanity tests (edge cases, rounding behavior, ledger invariants).

## Run
```bash
python demo_overflow_precision_practical.py
python exercises_starter.py
python run_tests.py
```

## Notes
- Python integers are arbitrary precision. We simulate i32/i64 limits to model
production constraints you’d face in C/C++/Rust/Go/Java.
- Float comparisons use a relative+absolute tolerance to work well both near
zero and far from it.
- For currency, prefer fixed-point or decimal.Decimal. Here we use int cents
(fast, easy to serialize). fixed_point.py keeps all balances as int.