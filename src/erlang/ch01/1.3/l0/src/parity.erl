-module(parity).
-export([is_even/1, is_odd/1, parity_bit/1]).

is_even(N) when is_integer(N) -> N rem 2 =:= 0.
is_odd(N)  when is_integer(N) -> N rem 2 =/= 0.

%% Return <<"even">> or <<"odd">> using the last bit
parity_bit(N) when is_integer(N) ->
    case (N band 1) of
        1 -> <<"odd">>;
        _ -> <<"even">>
    end.
