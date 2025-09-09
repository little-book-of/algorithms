# ch01 / 1.1 / l1 — Beyond Binary: Octal, Hex, and Two’s Complement (Erlang)

Examples for base representations and negative numbers.

## Files

* `src/demo_bases.erl` — print decimal, octal, and hex values.
* `src/twos_complement.erl` — encode/decode integers in two’s complement.
* `src/ch01_1_1_l1.erl` — demo and exercises runner.
* `test/ch01_1_1_l1_tests.erl` — unit tests (EUnit).
* `rebar.config` — project configuration.

## Build and Run

```
rebar3 compile
erl -pa _build/default/lib/*/ebin -s ch01_1_1_l1 main
```

## Test

```
rebar3 eunit
```
