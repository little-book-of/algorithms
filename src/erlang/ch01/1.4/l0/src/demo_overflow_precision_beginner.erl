-module(demo_overflow_precision_beginner).
-export([main/0]).

main() ->
    io:format("=== Integer overflow (8-bit) ===~n", []),
    io:format("255 + 1 (unsigned, wrap) -> ~p~n",
              [int_overflow_sim:add_unsigned_bits(255, 1, 8)]),
    io:format("127 + 1 (signed, wrap)   -> ~p~n",
              [int_overflow_sim:add_signed_bits(127, 1, 8)]),

    {R, CF, OF} = int_overflow_sim:add_with_flags_8(200, 100),
    io:format("200 + 100 -> result=~p (unsigned), CF=~p, OF=~p~n", [R, CF, OF]),
    io:format("Interpret result as signed: ~p~n",
              [int_overflow_sim:to_signed_bits(R, 8)]),

    io:format("~n=== Floating-point surprises ===~n", []),
    X = 0.1 + 0.2,
    io:format("0.1 + 0.2 = ~.17g~n", [X]),
    io:format("Direct equality with 0.3? ~p~n", [X =:= 0.3]),
    io:format("Using epsilon: ~p~n", [float_precision:almost_equal(X, 0.3, 1.0e-9)]),

    io:format("~nRepeat add 0.1 ten times:~n", []),
    S = float_precision:repeat_add(0.1, 10),
    io:format("sum = ~.17g, eq1=~p, almost=~p~n",
              [S, S =:= 1.0, float_precision:almost_equal(S, 1.0, 1.0e-9)]),

    io:format("~nMix large and small:~n", []),
    A = float_precision:mix_large_small(1.0e16, 1.0),
    io:format("1e16 + 1.0 = ~.17g (often unchanged)~n", [A]),

    io:format("~nNaive sum may drift slightly:~n", []),
    Nums = lists:duplicate(10, 0.1),
    io:format("sum_naive([0.1]*10) = ~.17g~n", [float_precision:sum_naive(Nums)]).
