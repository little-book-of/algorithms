-module(ch01_1_1_l1).
-export([main/0, exercises/0]).

main() ->
    demo_bases:run(),
    exercises().

exercises() ->
    %% Exercise 1: 100 in octal/hex
    N = 100,
    Oct = integer_to_list(N, 8),
    Hex = string:uppercase(integer_to_list(N, 16)),
    io:format("Exercise 1 (100): oct=~s hex=~s~n", [Oct, Hex]),

    %% Exercise 2: -7 in 8-bit two's complement
    Bits = 8,
    S = twos_complement:to(-7, Bits),
    io:format("Exercise 2 (-7 in ~p-bit): ~s~n", [Bits, S]),

    %% Exercise 3: verify 0xFF == 255 (parse hex string)
    VFF = list_to_integer("FF", 16),
    io:format("Exercise 3 (0xFF == 255?): ~p~n", [VFF =:= 255]),

    %% Exercise 4: parse "11111001" as 8-bit two's complement
    Val = twos_complement:from("11111001", 8),
    io:format("Exercise 4 (parse '11111001'): ~p~n", [Val]).
