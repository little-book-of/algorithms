-module(totient).
-export([phi/1]).

%% φ(N): count of 1..N coprime to N. Teaching version via trial factorization.
phi(N) when N =< 0 ->
    erlang:error(badarg);
phi(N) ->
    phi_(N, N, 2).

phi_(X, Result, P) when P*P > X ->
    case X > 1 of
        true  -> Result - (Result div X);
        false -> Result
    end;
phi_(X, Result, 2) ->
    case X rem 2 of
        0 -> phi_(strip(X, 2), Result - (Result div 2), 3);
        _ -> phi_(X, Result, 3)
    end;
phi_(X, Result, P) ->
    case X rem P of
        0 -> phi_(strip(X, P), Result - (Result div P), P+2);
        _ -> phi_(X, Result, P+2)
    end.

strip(X, P) when X rem P =:= 0 -> strip(X div P, P);
strip(X, _P) -> X.