-module(gcd_lcm).
-export([gcd/2, lcm/2]).

%% Euclid's algorithm; returns non-negative gcd.
gcd(A, B) ->
    gcd_abs(abs(A), abs(B)).

gcd_abs(A, 0) -> A;
gcd_abs(A, B) -> gcd_abs(B, A rem B).

%% lcm(0, B) = 0 by convention. Keep denominator positive.
lcm(0, _B) -> 0;
lcm(_A, 0) -> 0;
lcm(A, B) ->
    G = gcd(A, B),
    abs((A div G) * B).