-module(binary_ops).
-export([main/0]).

to_bin_str(X) when is_integer(X) ->
    lists:flatten(io_lib:format("~.2B", [X])).

main() ->
    X = 2#1011,  % 11
    Y = 2#0110,  % 6
    io:format("x = ~s (~p)~n", [to_bin_str(X), X]),
    io:format("y = ~s (~p)~n", [to_bin_str(Y), Y]),
    Sum = X + Y,
    io:format("x + y = ~s (~p)~n", [to_bin_str(Sum), Sum]),
    Diff = X - Y,
    io:format("x - y = ~s (~p)~n", [to_bin_str(Diff), Diff]).
