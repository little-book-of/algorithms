-module(ch01_1_1_l0_tests).
-include_lib("eunit/include/eunit.hrl").

dec_to_bin_test() ->
    ?assertEqual("0", dec_to_bin:convert(0)),
    ?assertEqual("1", dec_to_bin:convert(1)),
    ?assertEqual("10", dec_to_bin:convert(2)),
    ?assertEqual("101", dec_to_bin:convert(5)),
    ?assertEqual("10011", dec_to_bin:convert(19)),
    ?assertEqual("101010", dec_to_bin:convert(42)).

bin_to_dec_test() ->
    ?assertEqual(0, bin_to_dec:convert("0")),
    ?assertEqual(1, bin_to_dec:convert("1")),
    ?assertEqual(2, bin_to_dec:convert("10")),
    ?assertEqual(5, bin_to_dec:convert("101")),
    ?assertEqual(19, bin_to_dec:convert("10011")),
    ?assertEqual(42, bin_to_dec:convert("101010")).
