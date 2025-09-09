-module(exercises_starter).
-export([exercise1/1, exercise2/0, exercise3/0, exercise4/0, exercise5/0, main/0]).

exercise1(N) -> if parity:is_even(N) -> <<"even">>; true -> <<"odd">> end.
exercise2()  -> divisibility:is_divisible(91,7).
exercise3()  -> element(2, remainder:div_identity(100,9)).   %% remainder
exercise4()  -> remainder:week_shift(5,10).                   %% Sat + 10 -> Mon(1)
exercise5()  -> remainder:powmod(2,15,10).                    %% last digit of 2^15

main() ->
    io:format("Exercise 1 (42 parity): ~s~n", [exercise1(42)]),
    io:format("Exercise 2 (91 divisible by 7?): ~p~n", [exercise2()]),
    io:format("Exercise 3 (100 % 9): ~p~n", [exercise3()]),
    io:format("Exercise 4 (Sat+10): ~p~n", [exercise4()]),
    io:format("Exercise 5 (last digit of 2^15): ~p~n", [exercise5()]).
