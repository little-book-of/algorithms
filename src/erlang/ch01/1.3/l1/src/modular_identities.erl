-module(modular_identities).
-export([mod_norm/2, mod_add/3, mod_sub/3, mod_mul/3]).

%% Normalize X into [0, M-1] for M > 0.
mod_norm(X, M) when M =< 0 ->
    erlang:error(badarg);
mod_norm(X, M) ->
    R = X rem M,
    case R < 0 of
        true  -> R + M;
        false -> R
    end.

mod_add(A, B, M) ->
    mod_norm(mod_norm(A, M) + mod_norm(B, M), M).

mod_sub(A, B, M) ->
    mod_norm(mod_norm(A, M) - mod_norm(B, M), M).

mod_mul(A, B, M) ->
    mod_norm(mod_norm(A, M) * mod_norm(B, M), M).