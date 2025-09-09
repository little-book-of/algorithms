-module(karatsuba).
-export([mul/2, demo/0]).

%% Karatsuba multiplication for non-negative integers
mul(X, Y) when X < 10; Y < 10 ->
    X * Y;
mul(X, Y) ->
    N = max(bit_size(X), bit_size(Y)),
    M = N div 2,
    High1 = X bsr M,
    Low1  = X - (High1 bsl M),
    High2 = Y bsr M,
    Low2  = Y - (High2 bsl M),
    Z0 = mul(Low1, Low2),
    Z2 = mul(High1, High2),
    Z1 = mul(Low1+High1, Low2+High2) - Z0 - Z2,
    (Z2 bsl (2*M)) + (Z1 bsl M) + Z0.

bit_size(0) -> 0;
bit_size(N) -> bit_size(N, 0).
bit_size(0, Acc) -> Acc;
bit_size(N, Acc) -> bit_size(N bsr 1, Acc+1).

demo() ->
    A = 1234, B = 5678,
    Res = mul(A,B),
    io:format("Karatsuba result: ~p~n", [Res]),
    io:format("Expected: ~p~n", [A*B]).
