-module(efficiency_demo).
-export([main/0]).

%% For power-of-two modulus m = 2^k, n rem m == n band (m-1)
main() ->
    lists:foreach(
      fun(N) ->
          io:format("~p % 8 = ~p, bitmask = ~p~n", [N, N rem 8, N band 7])
      end,
      [5, 12, 20]).
