-module(demo_overflow_precision_practical).
-export([main/0]).

-define(I32_MAX, 2147483647).
-define(I64_MAX, 9223372036854775807).

main() ->
    io:format("=== Integers: checked / wrapping / saturating ===~n", []),
    io:format("int32 max: ~p~n", [?I32_MAX]),
    case int_arith:add_i32_checked(?I32_MAX, 1) of
        {error, overflow} -> io:format("checked add (i32): overflow trapped ✔~n", []);
        {ok, _}           -> io:format("checked add (i32): unexpected ok~n", [])
    end,
    io:format("wrapping add (i32): ~p~n", [int_arith:add_i32_wrapping(?I32_MAX, 1)]),
    io:format("saturating add (i32): ~p~n", [int_arith:add_i32_saturating(?I32_MAX, 1)]),

    io:format("wrapping add (i64): ~p~n", [int_arith:add_i64_wrapping(?I64_MAX-41, 100)]),
    io:format("saturating add (i64): ~p~n", [int_arith:add_i64_saturating(?I64_MAX-100, 200)]),

    io:format("~n=== Floats: compare, sum, epsilon ===~n", []),
    X = 0.1 + 0.2,
    io:format("0.1 + 0.2 = ~.17g~n", [X]),
    io:format("Direct equality with 0.3? ~p~n", [X =:= 0.3]),
    io:format("almost_equal(x, 0.3): ~p~n", [float_num:almost_equal(X, 0.3, 1.0e-12, 1.0e-12)]),

    N = 100000,
    Arr = [1.0/(I+1) || I <- lists:seq(0, N-1)],
    io:format("pairwise_sum: ~.15f~n", [float_num:pairwise_sum(Arr)]),
    io:format("kahan_sum   : ~.15f~n", [float_num:kahan_sum(Arr)]),
    io:format("machine epsilon: ~.20g~n", [float_num:machine_epsilon()]),
    io:format("ULP(1.0, nextafter_up(1.0)) = ~p~n", [float_num:ulp_diff(1.0, float_num:nextafter_up(1.0))]),

    io:format("~n=== Fixed-point money (int cents) ===~n", []),
    {ok, CA}   = fixed_point:parse_dollars_to_cents("10.00"),
    {ok, CDep} = fixed_point:parse_dollars_to_cents("0.05"),
    {ok, C335} = fixed_point:parse_dollars_to_cents("3.35"),
    A0 = fixed_point:ledger_new(CA),
    B0 = fixed_point:ledger_new(0),
    {ok, A1, B1} = fixed_point:ledger_transfer(A0, B0, C335),
    io:format("A: ~s B: ~s~n", [fixed_point:format_cents(A1#ledger.balance_cents),
                                 fixed_point:format_cents(B1#ledger.balance_cents)]),
    A2 = fixed_point:ledger_deposit(A1, CDep),
    io:format("A after 5 cents deposit: ~s~n", [fixed_point:format_cents(A2#ledger.balance_cents)]).
