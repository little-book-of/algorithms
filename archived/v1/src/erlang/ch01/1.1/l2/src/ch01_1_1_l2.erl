-module(ch01_1_1_l2).
-export([main/0, exercises/0]).

main() ->
    demo_floats:run(),
    exercises().

exercises() ->
    %% Ex1: fields of 1.0
    {S,E,M} = ieee754_utils:decompose(1.0),
    UExp = ieee754_utils:unbiased_exp(E),
    io:format("Ex1: sign=~B exp_raw=~B mantissa=0x~13.16.0B (unbiased=~B)~n", [S,E,M,UExp]),

    %% Ex2: ULP near 1.0
    U = ieee754_utils:ulp(1.0),
    NextUp = ieee754_utils:nextafter_pos_inf(1.0),
    io:format("Ex2: ULP(1.0)=~.17g nextUp=~.17g~n", [U, NextUp]),

    %% Ex3: classify representative values
    lists:foreach(
      fun({Name, Val}) ->
          io:format("Ex3: ~s -> ~s~n", [Name, atom_to_list(ieee754_utils:classify(Val))])
      end,
      [{"zero", 0.0},
       {"one",  1.0},
       {"+inf", ieee754_utils:build(0, (1 bsl 11) - 1, 0)},
       {"nan",  ieee754_utils:build(0, (1 bsl 11) - 1, 1)},
       {"first_subnormal", ieee754_utils:bits_to_float(1)}
      ]),

    %% Ex4: roundtrip bits for 3.5
    Bits = ieee754_utils:float_to_bits(3.5),
    Back = ieee754_utils:bits_to_float(Bits),
    io:format("Ex4: 3.5 bits=0x~16.16.0B back=~.17g~n", [Bits, Back]),

    %% Ex5: build +inf and a NaN explicitly
    Inf = ieee754_utils:build(0, (1 bsl 11) - 1, 0),
    NaN = ieee754_utils:build(0, (1 bsl 11) - 1, 1),
    io:format("Ex5: built +inf=~p, built nan==nan? ~p~n", [Inf, NaN =:= NaN]).
