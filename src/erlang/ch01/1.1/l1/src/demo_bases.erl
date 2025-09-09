%% Print decimal, octal, hex representations for a given integer.
-module(demo_bases).
-export([run/0]).

run() ->
    N = 42,
    Dec = integer_to_list(N),
    Oct = integer_to_list(N, 8),
    Hex = string:uppercase(integer_to_list(N, 16)),
    io:format("Decimal: ~s~n", [Dec]),
    io:format("Octal  : ~s~n", [Oct]),
    io:format("Hex    : ~s~n", [Hex]),

    %% Show literals via parsing strings (Erlang has only decimal literals).
    OctLit  = list_to_integer("52", 8),    % 52_8  = 42_10
    HexLit  = list_to_integer("2A", 16),   % 2A_16 = 42_10
    io:format("Octal literal 52 -> ~p~n", [OctLit]),
    io:format("Hex literal 2A -> ~p~n", [HexLit]).
