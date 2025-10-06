# 1.4 L1 — Overflow & Precision in Real Systems (Erlang)

Practical patterns that port well across languages:

- **Integers:** checked / wrapping / saturating add (i32/i64) and checked mul.
- **Floats:** relative+absolute tolerance compare, Kahan & pairwise summation,
  machine epsilon probe, ULP distance (with a tiny `nextafter_up/1`).
- **Money:** fixed-point `int64` cents utils and a tiny ledger.

## Run
rebar3 compile
erl -pa _build/default/lib/*/ebin -s demo_overflow_precision_practical main -s init stop
erl -pa _build/default/lib/*/ebin -s exercises_starter main -s init stop

## Test
rebar3 eunit

## Notes
- Erlang integers are arbitrary-precision; we **simulate** i32/i64 policies.
- “Checked” ops bound-check **before** arithmetic; “wrapping” uses masks
  and two’s-complement interpretation; “saturating” clamps to min/max.
- For currency, we keep balances in **int cents** (exact, easy to serialize).
