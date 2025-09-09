-module(bin_to_dec).
-export([convert/1]).

%% Convert a binary string like "1011" to integer.
convert(Str) when is_list(Str) ->
    lists:foldl(fun(Char, Acc) ->
        case Char of
            $0 -> Acc * 2;
            $1 -> Acc * 2 + 1;
            _  -> error(invalid_character)
        end
    end, 0, Str).
