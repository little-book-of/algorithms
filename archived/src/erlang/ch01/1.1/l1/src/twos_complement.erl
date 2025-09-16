%% Two's complement encode/decode for fixed bit-width.
-module(twos_complement).
-export([to/2, from/2]).

%% to(X, Bits) -> "bitstring" as list of chars '0'/'1' of length Bits.
%% X :: integer(), Bits :: 1..64 (you can raise if you want more).
to(X, Bits) when is_integer(X), is_integer(Bits), Bits >= 1, Bits =< 64 ->
    Mask = (1 bsl Bits) - 1,
    U = X band Mask,                                      % truncate to width
    Bin = integer_to_list(U, 2),
    PadLen = Bits - length(Bin),
    Padding = lists:duplicate(max(PadLen, 0), $0),
    Padding ++ Bin.

%% from(Str, Bits) -> signed integer according to two's complement with given Bits.
%% Str is a binary/iodata or string of '0'/'1'; Bits = length(Str) is typical.
from(Str0, Bits) when is_integer(Bits), Bits >= 1, Bits =< 64 ->
    Str = ensure_string(Str0),
    true = (length(Str) == Bits) orelse error({bad_width, {expected, Bits}, {got, length(Str)}}),
    Value = list_to_integer(Str, 2),
    Top   = 1 bsl (Bits - 1),
    case Value band Top of
        0 -> Value;                              % positive
        _ -> Value - (1 bsl Bits)                % negative (subtract 2^Bits)
    end.

ensure_string(S) when is_list(S) -> S;
ensure_string(B) when is_binary(B) -> binary_to_list(B).
