-module(extgcd_inv).
-export([extended_gcd/2, invmod/2]).

%% extended_gcd(A,B) -> {G,X,Y} with A*X + B*Y = G = gcd(A,B).
extended_gcd(A, B) ->
    extended_gcd_(A, B, 1, 0, 0, 1).

extended_gcd_(OldR, R, OldS, S, OldT, T) when R =:= 0 ->
    {OldR, OldS, OldT};
extended_gcd_(OldR, R, OldS, S, OldT, T) ->
    Q = OldR div R,
    extended_gcd_(R,
                  OldR - Q*R,
                  S,
                  OldS - Q*S,
                  T,
                  OldT - Q*T).

%% invmod(A, M) -> X in 0..M-1 with (A*X) rem M = 1; error if no inverse.
invmod(A, M) when M =< 0 ->
    erlang:error(badarg);
invmod(A, M) ->
    {G, X, _Y} = extended_gcd(A, M),
    case (G =:= 1) orelse (G =:= -1) of
        true  -> ((X rem M) + M) rem M;
        false -> erlang:error({no_inverse, {A, M}})
    end.