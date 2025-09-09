-module(float_precision).
-export([almost_equal/3,
         repeat_add/2,
         mix_large_small/2,
         sum_naive/1]).

%% almost_equal(X,Y,Eps) -> true if |X-Y| < Eps.
almost_equal(X, Y, Eps) ->
    abs(X - Y) < Eps.

%% repeat_add(Value, Times) -> add Value to 0.0 'Times' times.
repeat_add(Value, Times) when Times =< 0 -> 0.0;
repeat_add(Value, Times) ->
    repeat_add_(Value, Times, 0.0).

repeat_add_(Value, 0, Acc) -> Acc;
repeat_add_(Value, N, Acc) -> repeat_add_(Value, N-1, Acc + Value).

%% Adding tiny to huge may not change the number (limited precision).
mix_large_small(Large, Small) ->
    Large + Small.

%% sum_naive(List) -> naive left-to-right sum (shows accumulated error).
sum_naive(List) ->
    lists:foldl(fun(X, A) -> A + X end, 0.0, List).
