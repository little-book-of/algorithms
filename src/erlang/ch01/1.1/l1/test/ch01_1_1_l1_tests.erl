-module(ch01_1_1_l1_tests).
-include_lib("eunit/include/eunit.hrl").

demo_bases_test() ->
    ?assertEqual("52", integer_to_list(42, 8)),
    ?assertEqual("2A", string:uppercase(integer_to_list(42, 16))).

twos_to_test() ->
    ?assertEqual("11111001", twos_complement:to(-7, 8)),
    ?assertEqual("00000101", twos_complement:to(5, 8)).

twos_from_test() ->
    ?assertEqual(-7, twos_complement:from("11111001", 8)),
    ?assertEqual(5,  twos_complement:from("00000101", 8)).

exercise_hex_test() ->
    ?assertEqual(255, list_to_integer("FF", 16)).
