-module(demo_multiplication).
-export([main/0]).

naive_mul(X, Y) ->
    X * Y.

main() ->
    A = 1234, B = 5678,
    io:format("Naive multiplication: ~p~n", [naive_mul(A,B)]),
    io:format("Builtin multiplication: ~p~n", [A*B]).
