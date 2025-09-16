-module(ch01_1_1_l0).
-export([main/0, exercises/0]).

-include_lib("kernel/include/file.hrl").

main() ->
    N = 42,
    BinStr = dec_to_bin:convert(N),
    io:format("Decimal: ~p~n", [N]),
    io:format("Binary : ~s~n", [BinStr]),

    S = "1011",
    Val = bin_to_dec:convert(S),
    io:format("Binary string '~s' -> decimal: ~p~n", [S, Val]),

    exercises().

exercises() ->
    %% Exercise 1: 19 -> binary
    io:format("Exercise 1 (19 -> binary): ~s~n", [dec_to_bin:convert(19)]),

    %% Exercise 2: "10101" -> decimal
    io:format("Exercise 2 ('10101' -> dec): ~p~n", [bin_to_dec:convert("10101")]),

    %% Exercise 3: 27 -> binary
    io:format("Exercise 3 (27 -> binary): ~s~n", [dec_to_bin:convert(27)]),

    %% Exercise 4: verify "111111" == 63
    V = bin_to_dec:convert("111111"),
    io:format("Exercise 4: 111111 -> ~p == 63? ~p~n", [V, V == 63]),

    %% Exercise 5: why binary
    io:format("Exercise 5: Binary fits hardware with two stable states (0/1).~n").
