-module(remainder).
-export([div_identity/2, week_shift/2, powmod/3]).

%% Return {Q, R} s.t. N = D*Q + R (like Python divmod)
div_identity(N, D) when D =:= 0 ->
    erlang:error(badarg);
div_identity(N, D) ->
    {N div D, N rem D}.

%% Wrap days (0..6, 0=Mon); robust for negatives
week_shift(Start, Shift) ->
    M = 7,
    T = ((Start rem M) + M) rem M,
    S = ((Shift rem M) + M) rem M,
    (T + S) rem M.

%% Binary exponentiation: (Base^Exp) mod Mod
powmod(_Base, _Exp, 1) -> 0;
powmod(Base, Exp, Mod) ->
    powmod_( ((Base rem Mod) + Mod) rem Mod
           , Exp
           , Mod
           , 1 rem Mod ).

powmod_(_B, 0, _M, Acc) -> Acc;
powmod_(B, E, M, Acc) when (E band 1) =:= 1 ->
    powmod_((B*B) rem M, E bsr 1, M, (Acc*B) rem M);
powmod_(B, E, M, Acc) ->
    powmod_((B*B) rem M, E bsr 1, M, Acc).
