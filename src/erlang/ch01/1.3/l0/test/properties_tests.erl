-module(properties_tests).
-include_lib("eunit/include/eunit.hrl").

parity_test() ->
    ?assert(parity:is_even(10)),
    ?assert(parity:is_odd(7)),
    ?assertEqual(<<"odd">>, parity:parity_bit(11)),
    ?assertEqual(<<"even">>, parity:parity_bit(8)).

divisibility_test() ->
    ?assert(divisibility:is_divisible(12,3)),
    ?assertNot(divisibility:is_divisible(14,5)),
    ?assertEqual(4, divisibility:last_decimal_digit(1234)),
    ?assert(divisibility:divisible_by_10(120)).

remainder_and_clock_test() ->
    {Q,R} = remainder:div_identity(17,5),
    ?assertEqual({3,2}, {Q,R}),
    ?assertEqual(1, remainder:week_shift(5,10)),
    ?assertEqual(8, remainder:powmod(2,15,10)).

exercises_test() ->
    ?assertEqual(<<"even">>, exercises_starter:exercise1(42)),
    ?assert(exercises_starter:exercise2()),
    ?assertEqual(1, exercises_starter:exercise3()),
    ?assertEqual(1, exercises_starter:exercise4()),
    ?assertEqual(8, exercises_starter:exercise5()).
