-module(exercises_starter_tests).
-include_lib("eunit/include/eunit.hrl").

exercise1_test() ->
    ?assertEqual(31415926*27182818, exercises_starter:exercise1()).

exercise2_test() ->
    ?assertEqual(1, exercises_starter:exercise2()). %% 5^117 mod 19 = 1

exercise3_test() ->
    ?assert(exercises_starter:exercise3()).
