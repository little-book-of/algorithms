-module(demo_divmod).
-export([main/0]).

main() ->
    show(47, 5),
    show(23, 7),
    show(100, 9).

show(N, D) ->
    Q = N div D,
    R = N rem D,
    io:format("~p = ~p*~p + ~p~n", [N, D, Q, R]).
