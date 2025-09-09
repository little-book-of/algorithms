# ch01 / 1.3 / l0 — Simple Number Properties (Erlang)

Parity (even/odd), divisibility, and remainders with a 7-day “clock”.
Includes a small pow-mod helper for the last digit of 2^15.

## Files
- demo_properties.erl — quick tour: parity, divisibility, remainder/clock, last digit.
- parity.erl — even/odd helpers (mod and bit-based).
- divisibility.erl — divisibility helpers.
- remainder.erl — div identity, week wrap, powmod.
- exercises_starter.erl — small practice tasks.
- properties_tests.erl — EUnit sanity tests.

## Run
rebar3 compile
erl -pa _build/default/lib/*/ebin -s demo_properties main -s init stop
erl -pa _build/default/lib/*/ebin -s exercises_starter main -s init stop

## Test
rebar3 eunit
