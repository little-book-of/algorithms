# ch01 / 1.3 / l2 — Advanced Number Theory (Erlang)

Algorithms:
- Extended Euclid & modular inverse
- Euler’s totient φ(n) (trial factorization — good for teaching/small n)
- Miller–Rabin primality test (deterministic on 64-bit with fixed bases)
- Chinese Remainder Theorem (pairwise coprime moduli)

## Files
- demo_number_theory_advanced.erl — tour of invmod, φ, Miller–Rabin, CRT.
- extgcd_inv.erl — extended_gcd/2 and invmod/2.
- totient.erl — phi/1 via trial factorization.
- miller_rabin.erl — is_probable_prime/1 with fixed bases.
- crt.erl — CRTPair/2 and CRT/1 (pairwise coprime).
- exercises_starter.erl — practice tasks.
- number_theory_advanced_tests.erl — EUnit checks.

## Run
rebar3 compile
erl -pa _build/default/lib/*/ebin -s demo_number_theory_advanced main -s init stop
erl -pa _build/default/lib/*/ebin -s exercises_starter main -s init stop

## Test
rebar3 eunit