-module(exercises_starter).
-export([exercise1_policy/0, exercise2_compare_floats/0, exercise3_summation/0, exercise4_ledger/0, main/0]).

-define(I32_MAX, 2147483647).

exercise1_policy() ->
    Wrap = int_arith:add_i32_wrapping(?I32_MAX, 1),
    Checked = case int_arith:add_i32_checked(?I32_MAX, 1) of
        {error, overflow} -> overflow; {ok, _} -> ok end,
    Sat = int_arith:add_i32_saturating(?I32_MAX, 1),
    {Wrap, Checked, Sat}.

exercise2_compare_floats() ->
    X = 0.1 + 0.2,
    {X, X =:= 0.3, float_num:almost_equal(X, 0.3, 1.0e-12, 1.0e-12)}.

exercise3_summation() ->
    N = 100000,
    Xs = [1.0/(I+1) || I <- lists:seq(0, N-1)],
    {float_num:pairwise_sum(Xs), float_num:kahan_sum(Xs)}.

exercise4_ledger() ->
    {ok, A0} = fixed_point:parse_dollars_to_cents("12.34"),
    {ok, B0} = fixed_point:parse_dollars_to_cents("0.00"),
    {ok, Amt} = fixed_point:parse_dollars_to_cents("0.34"),
    LA = fixed_point:ledger_new(A0),
    LB = fixed_point:ledger_new(B0),
    {ok, LA1, LB1} = fixed_point:ledger_transfer(LA, LB, Amt),
    {fixed_point:format_cents(LA1#ledger.balance_cents), fixed_point:format_cents(LB1#ledger.balance_cents)}.

main() ->
    {W,Chk,Sat} = exercise1_policy(),
    io:format("Exercise 1 (policy): wrap=~p checked=~p sat=~p~n", [W,Chk,Sat]),
    {Sum,Eq1,Almost} = exercise2_compare_floats(),
    io:format("Exercise 2 (float compare): sum=~.17g eq=~p almost=~p~n", [Sum,Eq1,Almost]),
    {P,K} = exercise3_summation(),
    io:format("Exercise 3 (pairwise vs kahan): ~.15f ~.15f~n", [P,K]),
    {AS,BS} = exercise4_ledger(),
    io:format("Exercise 4 (ledger): ~s ~s~n", [AS,BS]).
