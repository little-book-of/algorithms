-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/0, main/0]).

exercise1() ->
    karatsuba:mul(31415926, 27182818).

exercise2() ->
    modexp:modexp(5, 117, 19).

exercise3() ->
    A = 12345, B = 67890,
    karatsuba:mul(A,B) =:= A*B.

main() ->
    io:format("Exercise 1 (Karatsuba 31415926*27182818): ~p~n", [exercise1()]),
    io:format("Exercise 2 (modexp 5^117 mod 19): ~p~n", [exercise2()]),
    io:format("Exercise 3 (check Karatsuba correctness): ~p~n", [exercise3()]).
