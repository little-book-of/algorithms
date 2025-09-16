-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/2, exercise4/1, exercise5/1, main/0]).

exercise1() -> {100 div 9, 100 rem 9}.
exercise2() -> {123 div 11, 123 rem 11}.
exercise3(N, D) ->
    Q = N div D, R = N rem D,
    N =:= D*Q + R.
exercise4(N) -> N band 15.  %% N % 16
exercise5(N) -> (N rem 7) =:= 0.

main() ->
    {Q1,R1} = exercise1(),
    io:format("Exercise 1 (100//9,100%9): ~p ~p~n", [Q1,R1]),
    {Q2,R2} = exercise2(),
    io:format("Exercise 2 (123//11,123%11): ~p ~p~n", [Q2,R2]),
    io:format("Exercise 3 check (200,23): ~p~n", [exercise3(200,23)]),
    io:format("Exercise 4 (37 % 16 via bitmask): ~p~n", [exercise4(37)]),
    io:format("Exercise 5 (35 divisible by 7?): ~p~n", [exercise5(35)]).
