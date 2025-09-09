-module(crt).
-export([crt_pair/2, crt/1]).

%% crt_pair({A1,M1}, {A2,M2}) with coprime moduli -> {X, M1*M2}
crt_pair({A1,M1}, {A2,M2}) when M1 =< 0; M2 =< 0 ->
    erlang:error(badarg);
crt_pair({A1,M1}, {A2,M2}) ->
    %% Require gcd(M1,M2)=1
    {G, _, _} = extgcd_inv:extended_gcd(M1, M2),
    case (G =:= 1) orelse (G =:= -1) of
        false -> erlang:error({non_coprime_moduli, {M1, M2}});
        true  ->
            Inv = extgcd_inv:invmod(M1 rem M2, M2),
            K   = (((A2 - A1) rem M2) + M2) rem M2,
            K1  = (K * Inv) rem M2,
            M   = M1 * M2,
            X   = ((A1 + K1 * M1) rem M + M) rem M,
            {X, M}
    end.

%% Fold many congruences with pairwise coprime moduli
crt([]) -> erlang:error(badarg);
crt([{A,M}|Rest]) ->
    lists:foldl(fun({A2,M2}, {X,M1}) -> crt_pair({X,M1}, {A2,M2}) end,
                {A,M}, Rest).