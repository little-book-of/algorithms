-module(miller_rabin).
-export([is_probable_prime/1]).

%% Deterministic for 64-bit with bases {2,3,5,7,11,13,17}.
%% Works for larger ints as a probabilistic test.

is_probable_prime(N) when N < 2 -> false;
is_probable_prime(N) ->
    Small = [2,3,5,7,11,13,17,19,23,29],
    case lists:member(N, Small) of
        true  -> true;
        false ->
            case has_small_factor(N, Small) of
                true  -> false;
                false ->
                    {D,S} = decompose(N-1),
                    Bases = [2,3,5,7,11,13,17],
                    not lists:any(fun(A0) ->
                        A = A0 rem N,
                        A =/= 0 andalso try_composite(A, D, N, S)
                    end, Bases)
            end
    end.

has_small_factor(_N, []) -> false;
has_small_factor(N, [P|Ps]) ->
    if N rem P =:= 0 -> true; true -> has_small_factor(N, Ps) end.

%% n-1 = d * 2^s with d odd
decompose(N1) ->
    decompose_(N1, 0).
decompose_(D, S) when (D band 1) =:= 1 -> {D, S};
decompose_(D, S) -> decompose_(D bsr 1, S+1).

%% Fast powmod using square-and-multiply
powmod(_Base, 0, _Mod) -> 1;
powmod(Base, Exp, Mod) when (Exp band 1) =:= 1 ->
    (Base * powmod((Base*Base) rem Mod, Exp bsr 1, Mod)) rem Mod;
powmod(Base, Exp, Mod) ->
    powmod((Base*Base) rem Mod, Exp bsr 1, Mod).

try_composite(A, D, N, S) ->
    X0 = powmod(A, D, N),
    case (X0 =:= 1) orelse (X0 =:= N-1) of
        true  -> false;
        false -> loop_squares(X0, N, S-1)
    end.

loop_squares(_X, _N, 0) -> true;  %% composite witness
loop_squares(X, N, K) ->
    X2 = (X*X) rem N,
    case X2 =:= N-1 of
        true  -> false;
        false -> loop_squares(X2, N, K-1)
    end.