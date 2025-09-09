-module(overflow_precision_practical_tests).
-include_lib("eunit/include/eunit.hrl").

i32_ops_test() ->
    ?assertMatch({error, overflow}, int_arith:add_i32_checked(2147483647, 1)),
    ?assertEqual(-2147483648, int_arith:add_i32_wrapping(2147483647, 1)),
    ?assertEqual(2147483647, int_arith:add_i32_saturating(2147483647, 1)),
    ?assertMatch({error, overflow}, int_arith:mul_i32_checked(100000, 100000)),
    {ok, P} = int_arith:mul_i32_checked(46341, 46341),
    ?assertEqual(2147488281, P).

i64_ops_test() ->
    ?assertMatch({error, overflow}, int_arith:add_i64_checked(9223372036854775807, 1)),
    ?assertEqual(9223372036854775807, int_arith:add_i64_saturating(9223372036854775807, 1)),
    ?assertEqual(-9223372036854775808, int_arith:add_i64_wrapping(9223372036854775807, 1)),
    ?assertMatch({error, overflow}, int_arith:mul_i64_checked(9223372036854775807, 2)).

float_utils_test() ->
    X = 0.1 + 0.2,
    ?assertNot(X =:= 0.3),
    ?assert(float_num:almost_equal(X, 0.3, 1.0e-12, 1.0e-12)),
    N = 50000,
    Xs = [1.0/(I+1) || I <- lists:seq(0, N-1)],
    P = float_num:pairwise_sum(Xs),
    K = float_num:kahan_sum(Xs),
    ?assert(abs(P - K) < 1.0e-10),
    Eps = float_num:machine_epsilon(),
    ?assert(1.0 + Eps =/= 1.0),
    Next = float_num:nextafter_up(1.0),
    ?assertEqual(1, float_num:ulp_diff(1.0, Next)).

money_test() ->
    ?assertMatch({ok, 1234}, fixed_point:parse_dollars_to_cents("12.34")),
    ?assertMatch({ok, -7}, fixed_point:parse_dollars_to_cents("-0.07")),
    %% truncate beyond 2 decimals
    ?assertMatch({ok, 123}, fixed_point:parse_dollars_to_cents("1.239")),
    {ok, Ten} = fixed_point:parse_dollars_to_cents("10.00"),
    {ok, C335} = fixed_point:parse_dollars_to_cents("3.35"),
    A0 = fixed_point:ledger_new(Ten),
    B0 = fixed_point:ledger_new(0),
    {ok, A1, B1} = fixed_point:ledger_transfer(A0, B0, C335),
    ?assertEqual("6.65", fixed_point:format_cents(A1#ledger.balance_cents)),
    ?assertEqual("3.35", fixed_point:format_cents(B1#ledger.balance_cents)).

exercises_test() ->
    {Wrap,Chk,Sat} = exercises_starter:exercise1_policy(),
    ?assertEqual(-2147483648, Wrap),
    ?assertEqual(overflow, Chk),
    ?assertEqual(2147483647, Sat),
    {Sum, Eq, Almost} = exercises_starter:exercise2_compare_floats(),
    ?assertNot(Eq), ?assert(Almost),
    {P,K} = exercises_starter:exercise3_summation(),
    ?assert(abs(P-K) < 1.0e-10),
    {AS,BS} = exercises_starter:exercise4_ledger(),
    ?assertEqual("12.00", AS), ?assertEqual("0.34", BS).
