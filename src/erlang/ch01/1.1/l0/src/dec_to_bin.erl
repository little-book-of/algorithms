-module(dec_to_bin).
-export([convert/1]).

%% Convert a non-negative integer to its binary string (as list of chars).
convert(0) -> "0";
convert(N) when is_integer(N), N >= 0 ->
    convert(N, []).
    
convert(0, Acc) -> Acc;
convert(N, Acc) ->
    Bit = integer_to_list(N rem 2),
    convert(N div 2, [Bit | Acc]).
