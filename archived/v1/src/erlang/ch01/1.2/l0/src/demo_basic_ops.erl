-module(demo_basic_ops).
-export([main/0]).

main() ->
    A = 478, B = 259,
    io:format("Addition: ~p + ~p = ~p~n", [A, B, A + B]),

    A1 = 503, B1 = 78,
    io:format("Subtraction: ~p - ~p = ~p~n", [A1, B1, A1 - B1]),

    A2 = 214, B2 = 3,
    io:format("Multiplication: ~p * ~p = ~p~n", [A2, B2, A2 * B2]),

    N = 47, D = 5,
    Q = N div D, R = N rem D,
    io:format("Division: ~p / ~p -> quotient ~p, remainder ~p~n", [N, D, Q, R]).
