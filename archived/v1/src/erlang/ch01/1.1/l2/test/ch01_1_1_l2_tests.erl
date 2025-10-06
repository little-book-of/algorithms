-module(ch01_1_1_l2_tests).
-include_lib("eunit/include/eunit.hrl").

decompose_one_test() ->
    {S,E,M} = ieee754_utils:decompose(1.0),
    ?assertEqual(0, S),
    ?assertEqual(1023, E),
    ?assertEqual(0, M),
    ?assertEqual(0, ieee754_utils:unbiased_exp(E)).

roundtrip_test() ->
    X = 3.5,
    Bits = ieee754_utils:float_to_bits(X),
    Y = ieee754_utils:bits_to_float(Bits),
    ?assertEqual(X, Y).

classify_test() ->
    Tiny = ieee754_utils:bits_to_float(1),
    ?assertEqual(zero, ieee754_utils:classify(0.0)),
    ?assertEqual(normal, ieee754_utils:classify(1.0)),
    ?assertEqual(subnormal, ieee754_utils:classify(Tiny)),
    ?assertEqual(inf, ieee754_utils:classify(ieee754_utils:build(0, (1 bsl 11) - 1, 0))),
    ?assertEqual(nan, ieee754_utils:classify(ieee754_utils:build(0, (1 bsl 11) - 1, 1))).

ulp_test() ->
    UL = ieee754_utils:ulp(1.0),
    NextUp = ieee754_utils:nextafter_pos_inf(1.0),
    ?assertEqual(NextUp - 1.0, UL).
