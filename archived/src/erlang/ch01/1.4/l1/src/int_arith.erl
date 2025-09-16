%% Integer arithmetic policies: checked / wrapping / saturating (i32/i64) + checked mul.
-module(int_arith).
-export([
    add_i32_checked/2, add_i32_wrapping/2, add_i32_saturating/2, mul_i32_checked/2,
    add_i64_checked/2, add_i64_wrapping/2, add_i64_saturating/2, mul_i64_checked/2
]).

-define(INT32_MIN, -2147483648).
-define(INT32_MAX,  2147483647).
-define(INT64_MIN, -9223372036854775808).
-define(INT64_MAX,  9223372036854775807).

%% --- helpers ---

to_signed_bits(Value, Bits) ->
    Mask = (1 bsl Bits) - 1,
    V = Value band Mask,
    Sign = 1 bsl (Bits - 1),
    case (V band Sign) =/= 0 of
        true  -> V - (1 bsl Bits);
        false -> V
    end.

wrap_add_bits(A, B, Bits) ->
    to_signed_bits((A + B) band ((1 bsl Bits) - 1), Bits).

%% --- i32 ---

add_i32_checked(A, B) ->
    S = A + B,
    if S < ?INT32_MIN orelse S > ?INT32_MAX -> {error, overflow};
       true                                 -> {ok, S}
    end.

add_i32_wrapping(A, B) ->
    wrap_add_bits(A, B, 32).

add_i32_saturating(A, B) ->
    S = A + B,
    if S > ?INT32_MAX -> ?INT32_MAX;
       S < ?INT32_MIN -> ?INT32_MIN;
       true           -> S
    end.

mul_i32_checked(A, B) ->
    P = A * B,
    if P < ?INT32_MIN orelse P > ?INT32_MAX -> {error, overflow};
       true                                 -> {ok, P}
    end.

%% --- i64 ---

add_i64_checked(A, B) ->
    S = A + B,
    if S < ?INT64_MIN orelse S > ?INT64_MAX -> {error, overflow};
       true                                 -> {ok, S}
    end.

add_i64_wrapping(A, B) ->
    wrap_add_bits(A, B, 64).

add_i64_saturating(A, B) ->
    S = A + B,
    if S > ?INT64_MAX -> ?INT64_MAX;
       S < ?INT64_MIN -> ?INT64_MIN;
       true           -> S
    end.

mul_i64_checked(A, B) ->
    P = A * B,
    if P < ?INT64_MIN orelse P > ?INT64_MAX -> {error, overflow};
       true                                 -> {ok, P}
    end.
