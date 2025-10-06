-module(modexp).
-export([modexp/3, demo/0]).

%% Modular exponentiation using square-and-multiply
modexp(A, B, M) ->
    modexp(A rem M, B, M, 1).

modexp(_, 0, _, Result) -> Result;
modexp(Base, Exp, M, Result) when (Exp band 1) =:= 1 ->
    modexp((Base*Base) rem M, Exp bsr 1, M, (Result*Base) rem M);
modexp(Base, Exp, M, Result) ->
    modexp((Base*Base) rem M, Exp bsr 1, M, Result).

demo() ->
    io:format("modexp(7,128,13) = ~p~n", [modexp(7,128,13)]).
