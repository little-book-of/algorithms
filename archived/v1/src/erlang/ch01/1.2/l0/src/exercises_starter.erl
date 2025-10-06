-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/0, exercise4/0, main/0]).

exercise1() -> 326 + 589.
exercise2() -> 704 - 259.
exercise3() -> 38 * 12.
exercise4() -> {123 div 7, 123 rem 7}.

main() ->
    io:format("Exercise 1 (326+589): ~p~n", [exercise1()]),
    io:format("Exercise 2 (704-259): ~p~n", [exercise2()]),
    io:format("Exercise 3 (38*12): ~p~n", [exercise3()]),
    {Q, R} = exercise4(),
    io:format("Exercise 4 (123//7, 123%7): ~p ~p~n", [Q, R]).
