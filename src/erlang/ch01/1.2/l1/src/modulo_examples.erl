-module(modulo_examples).
-export([hashing_example/2, cyclic_day/2, modexp/3, main/0]).

%% Simple hash bucket mapping
hashing_example(Key, Size) ->
    Key rem Size.

%% Cyclic wrap-around for weekdays (0–6)
cyclic_day(Start, Shift) ->
    (Start + Shift) rem 7.

%% Modular exponentiation (square-and-multiply)
modexp(A, B, M) ->
    modexp(A rem M, B, M, 1).

modexp(_, 0, _, Result) -> Result;
modexp(Base, Exp, M, Result) when Exp band 1 =:= 1 ->
    modexp((Base * Base) rem M, Exp bsr 1, M, (Result * Base) rem M);
modexp(Base, Exp, M, Result) ->
    modexp((Base * Base) rem M, Exp bsr 1, M, Result).

main() ->
    io:format("Hashing: key=1234, size=10 -> ~p~n", [hashing_example(1234, 10)]),
    io:format("Cyclic: Saturday(5)+4 -> ~p~n", [cyclic_day(5, 4)]),
    io:format("modexp(7,128,13) = ~p~n", [modexp(7, 128, 13)]).
