-module(demo_properties).
-export([main/0]).

main() ->
    io:format("Parity:~n", []),
    lists:foreach(
      fun(N) ->
          io:format("  ~p is_even? ~p  is_odd? ~p  (bit) ~s~n",
                    [N, parity:is_even(N), parity:is_odd(N), parity:parity_bit(N)])
      end,
      [10, 7]),

    io:format("~nDivisibility:~n", []),
    io:format("  12 divisible by 3? ~p~n", [divisibility:is_divisible(12,3)]),
    io:format("  14 divisible by 5? ~p~n", [divisibility:is_divisible(14,5)]),

    io:format("~nRemainders & identity:~n", []),
    {Q,R} = remainder:div_identity(17,5),
    io:format("  17 = 5*~p + ~p~n", [Q,R]),

    io:format("~nClock (7-day week):~n", []),
    io:format("  Saturday(5) + 10 days -> ~p~n", [remainder:week_shift(5,10)]),

    io:format("~nLast digit of 2^15:~n  ~p~n", [remainder:powmod(2,15,10)]).
