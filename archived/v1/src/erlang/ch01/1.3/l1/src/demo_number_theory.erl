-module(demo_number_theory).
-export([main/0]).

main() ->
    %% GCD / LCM
    A = 252, B = 198,
    io:format("gcd(~p, ~p) = ~p~n", [A, B, gcd_lcm:gcd(A,B)]),
    io:format("lcm(12, 18) = ~p~n", [gcd_lcm:lcm(12, 18)]),

    %% Modular identities
    X = 123, Y = 456, M = 7,
    io:format("(x+y)%%m vs mod_add: ~p ~p~n", [(X+Y) rem M, modular_identities:mod_add(X,Y,M)]),
    io:format("(x*y)%%m vs mod_mul: ~p ~p~n", [(X*Y) rem M, modular_identities:mod_mul(X,Y,M)]),

    %% Fraction reduction
    {Num,Den} = fractions_utils:reduce_fraction({84,126}),
    io:format("reduce_fraction(84,126) = ~p/~p~n", [Num, Den]).