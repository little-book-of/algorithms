-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/0, exercise4/0, main/0]).

%% 1) 4-bit counter: start at 14, add 1 three times.
exercise1() ->
    A = int_overflow_sim:add_unsigned_bits(14, 1, 4), % 15
    B = int_overflow_sim:add_unsigned_bits(A, 1, 4),  % 0
    C = int_overflow_sim:add_unsigned_bits(B, 1, 4),  % 1
    {A, B, C}.

%% 2) Ten times 0.1 vs 1.0 (exact? almost-equal?)
exercise2() ->
    S = float_precision:repeat_add(0.1, 10),
    {S, S =:= 1.0, float_precision:almost_equal(S, 1.0, 1.0e-9)}.

%% 3) 8-bit signed addition demos: 127+1 and -1+(-1)
exercise3() ->
    A = int_overflow_sim:add_signed_bits(127, 1, 8),  % -128
    B = int_overflow_sim:add_signed_bits(-1, -1, 8), % -2
    {A, B}.

%% 4) Unsigned 16-bit wrap for 65535 + 1 -> 0
exercise4() ->
    int_overflow_sim:add_unsigned_bits(65535, 1, 16).

main() ->
    {A,B,C} = exercise1(),
    io:format("Exercise 1 (4-bit sequence from 14): ~p ~p ~p~n", [A,B,C]),
    {Sum, Eq1, Almost} = exercise2(),
    io:format("Exercise 2 (ten*0.1): sum=~.17g eq1=~p almost=~p~n", [Sum, Eq1, Almost]),
    {X,Y} = exercise3(),
    io:format("Exercise 3 (signed adds): ~p ~p~n", [X,Y]),
    io:format("Exercise 4 (u16 65535+1): ~p~n", [exercise4()]).
