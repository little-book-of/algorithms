# ch01 / 1.3 / l1 — Classical Number Theory Tools (Erlang)

Euclid's GCD, LCM via GCD, modular arithmetic identities, and a tiny fraction reducer.

## Files
- demo_number_theory.erl — quick tour of gcd/lcm, modular identities, and fraction reduction.
- gcd_lcm.erl — Euclid's algorithm and lcm(A,B) = A*B / gcd(A,B).
- modular_identities.erl — mod_add/mod_sub/mod_mul with normalization into [0, M-1].
- fractions_utils.erl — reduce_fraction({N,D}) with canonical sign (D > 0).
- exercises_starter.erl — tasks mirroring the text.
- number_theory_tests.erl — EUnit sanity checks.

## Run
rebar3 compile
erl -pa _build/default/lib/*/ebin -s demo_number_theory main -s init stop
erl -pa _build/default/lib/*/ebin -s exercises_starter main -s init stop

## Test
rebar3 eunit