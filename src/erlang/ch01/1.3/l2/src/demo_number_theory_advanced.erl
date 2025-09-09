-module(demo_number_theory_advanced).
-export([main/0]).

main() ->
    %% Extended Euclid / inverse
    {G, X, Y} = extgcd_inv:extended_gcd(240, 46),
    io:format("extended_gcd(240,46) -> ~p~n", [{G,X,Y}]),
    io:format("invmod(3,7) = ~p~n", [extgcd_inv:invmod(3,7)]),

    %% Totient
    io:format("phi(10) = ~p~n", [totient:phi(10)]),
    io:format("phi(36) = ~p~n", [totient:phi(36)]),

    %% Miller–Rabin
    lists:foreach(
      fun(N) ->
          io:format("is_probable_prime(~p) = ~p~n", [N, miller_rabin:is_probable_prime(N)])
      end,
      [97, 561, 1105, 2147483647]),

    %% CRT: x≡2 (mod 3), x≡3 (mod 5), x≡2 (mod 7) -> 23 mod 105
    {X12, M12} = crt:crt_pair({2,3}, {3,5}),
    {Res, Mod} = crt:crt_pair({X12, M12}, {2,7}),
    io:format("CRT -> ~p mod ~p~n", [Res, Mod]).