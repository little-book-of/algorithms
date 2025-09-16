# ch01 / 1.1 / l0 — Decimal and Binary Basics (Erlang)

Examples for decimal ↔ binary basics.

## Files

* `src/dec_to_bin.erl` — convert decimal (non-negative integer) → binary string.
* `src/bin_to_dec.erl` — convert binary string → decimal (integer).
* `src/ch01_1_1_l0.erl` — demo and exercises runner.
* `test/ch01_1_1_l0_tests.erl` — unit tests (EUnit).
* `rebar.config` — project configuration.

## Build and Run

```
rebar3 compile
erl -pa _build/default/lib/*/ebin -s ch01_1_1_l0 main
```

## Test

```
rebar3 eunit
```
