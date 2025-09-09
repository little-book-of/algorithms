-module(number_theory_tests).
-include_lib("eunit/include/eunit.hrl").

gcd_lcm_test() ->
    ?assertEqual(2, gcd_lcm:gcd(20,14)),
    ?assertEqual(18, gcd_lcm:gcd(252,198)),
    ?assertEqual(36, gcd_lcm:lcm(12,18)),
    ?assertEqual(0, gcd_lcm:lcm(0,5)).

modular_identities_test() ->
    A = 123, B = 456, M = 7,
    ?assertEqual(((A rem M) + (B rem M)) rem M, modular_identities:mod_add(A,B,M)),
    ?assertEqual(((A rem M) - (B rem M)) rem M |> norm(M),
                 modular_identities:mod_sub(A,B,M)),
    ?assertEqual(((A rem M) * (B rem M)) rem M, modular_identities:mod_mul(A,B,M)),
    ?assertEqual(((37 rem 12) + (85 rem 12)) rem 12, modular_identities:mod_add(37,85,12)).

fraction_reduce_test() ->
    ?assertEqual({2,3},  fractions_utils:reduce_fraction({84,126})),
    ?assertEqual({-3,4}, fractions_utils:reduce_fraction({-6,8})),
    ?assertEqual({-3,4}, fractions_utils:reduce_fraction({6,-8})),
    ?assertEqual({3,4},  fractions_utils:reduce_fraction({-6,-8})).

exercises_test() ->
    ?assertEqual(18, exercises_starter:exercise1()),
    ?assertEqual(36, exercises_starter:exercise2()),
    {Lhs, Rhs} = exercises_starter:exercise3(),
    ?assertEqual(Lhs, Rhs),
    ?assertEqual({2,3}, exercises_starter:exercise4()),
    ?assertEqual(36, exercises_starter:exercise5()).

%% helper: normalize Erlang's rem result into [0,M-1]
norm(R, M) when R < 0 -> R + M;
norm(R, _M) -> R.