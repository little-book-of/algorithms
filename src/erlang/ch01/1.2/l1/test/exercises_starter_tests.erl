-module(exercises_starter_tests).
-include_lib("eunit/include/eunit.hrl").

exercise1_test() ->
    ?assertEqual({11,1}, exercises_starter:exercise1()).

exercise2_test() ->
    ?assertEqual({11,2}, exercises_starter:exercise2()).

exercise3_test() ->
    ?assert(exercises_starter:exercise3(200,23)).

exercise4_test() ->
    ?assertEqual(5, exercises_starter:exercise4(37)).

exercise5_test() ->
    ?assert(exercises_starter:exercise5(35)).
