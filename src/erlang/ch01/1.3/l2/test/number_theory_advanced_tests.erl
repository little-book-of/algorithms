-module(number_theory_advanced_tests).
-include_lib("eunit/include/eunit.hrl").

extended_gcd_inv_test() ->
    {G,X,Y} = extgcd_inv:extended_gcd(240,46),
    ?assertEqual(2, G),
    ?assertEqual(2, 240*X + 46*Y),
    ?assertEqual(5, extgcd_inv:invmod(3,7)),
    ?assertException(error, {no_inverse, _}, fun() -> extgcd_inv:invmod(6,9) end).

phi_test() ->
    ?assertEqual(1,  totient:phi(1)),
    ?assertEqual(4,  totient:phi(10)),
    ?assertEqual(12, totient:phi(36)),
    ?assertEqual(96, totient:phi(97)).

miller_rabin_test() ->
    %% primes
    lists:foreach(fun(P) -> ?assert(miller_rabin:is_probable_prime(P)) end,
                  [2,3,5,7,11,97,569,2147483647]),
    %% composites
    lists:foreach(fun(C) -> ?assertNot(miller_rabin:is_probable_prime(C)) end,
                  [1,4,6,8,9,10,12,100]),
    %% Carmichael numbers should be rejected
    lists:foreach(fun(C) -> ?assertNot(miller_rabin:is_probable_prime(C)) end,
                  [561,1105,1729,2465]).

crt_test() ->
    {X, M} = crt:crt([{2,3},{3,5},{2,7}]),
    ?assertEqual(105, M),
    ?assertEqual(2, X rem 3),
    ?assertEqual(3, X rem 5),
    ?assertEqual(2, X rem 7).

exercises_test() ->
    ?assertEqual(2, exercises_starter:exercise1()),
    {Phi10, Ok} = exercises_starter:exercise2(),
    ?assertEqual(4, Phi10), ?assert(Ok),
    ?assertEqual(1, exercises_starter:exercise3()),
    ?assertEqual([97,569], exercises_starter:exercise4()),
    {Sol, Mod} = exercises_starter:exercise5(),
    ?assertEqual(52, Sol), ?assertEqual(140, Mod).