-module(divisibility).
-export([is_divisible/2, last_decimal_digit/1, divisible_by_10/1]).

is_divisible(_A, 0) -> false;  %% undefined; keep simple at L0
is_divisible(A, B) when is_integer(A), is_integer(B) ->
    A rem B =:= 0.

last_decimal_digit(N) when is_integer(N) ->
    abs(N) rem 10.

divisible_by_10(N) ->
    last_decimal_digit(N) =:= 0.
