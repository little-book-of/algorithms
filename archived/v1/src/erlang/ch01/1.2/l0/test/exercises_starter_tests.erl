-module(exercises_starter_tests).
-include_lib("eunit/include/eunit.hrl").

exercise1_test() ->
    ?assertEqual(915, exercises_starter:exercise1()).

exercise2_test() ->
    ?assertEqual(445, exercises_starter:exercise2()).

exercise3_test() ->
    ?assertEqual(456, exercises_starter:exercise3()).

exercise4_test() ->
    {Q, R} = exercises_starter:exercise4(),
    ?assertEqual({17, 4}, {Q, R}).
