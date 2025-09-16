-module(demo_floats).
-export([run/0]).

run() ->
    %% Rounding surprise
    A = 0.1 + 0.2,
    io:format("0.1 + 0.2 = ~.17g~n", [A]),
    io:format("Equal to 0.3? ~p~n", [A =:= 0.3]),

    %% Special values via classification
    PInf = ieee754_utils:build(0, (1 bsl 11) - 1, 0),
    NaN  = ieee754_utils:build(0, (1 bsl 11) - 1, 1),
    io:format("Infinity: ~p~n", [PInf]),
    io:format("NaN == NaN? ~p~n", [NaN =:= NaN]),

    %% ULP near 1.0 and nextafter(+inf)
    U = ieee754_utils:ulp(1.0),
    NextUp = ieee754_utils:nextafter_pos_inf(1.0),
    NextDown = ieee754_utils:bits_to_float(ieee754_utils:float_to_bits(1.0) - 1),
    io:format("ULP(1.0): ~.17g~n", [U]),
    io:format("nextafter(1.0, +inf): ~.17g~n", [NextUp]),
    io:format("nextafter(1.0, -inf): ~.17g~n", [NextDown]),
    io:format("ULP around 1.0 (nextUp - 1.0): ~.17g~n", [NextUp - 1.0]),

    %% Subnormal example (first positive subnormal)
    Tiny = ieee754_utils:bits_to_float(1),
    io:format("First subnormal > 0: ~.17g~n", [Tiny]),
    io:format("Classify tiny: ~s~n", [atom_to_list(ieee754_utils:classify(Tiny))]),

    %% Decompose a few values
    lists:foreach(fun(X) ->
        Str = ieee754_utils:components_pretty(X),
        io:format("~s~n", [Str])
    end, [1.0, 0.0, PInf, NaN]).
