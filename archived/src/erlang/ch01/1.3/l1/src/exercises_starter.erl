-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/0, exercise4/0, exercise5/0, main/0]).

exercise1() -> gcd_lcm:gcd(252, 198).
exercise2() -> gcd_lcm:lcm(12, 18).
exercise3() ->
    Lhs = (37 + 85) rem 12,
    Rhs = modular_identities:mod_add(37, 85, 12),
    {Lhs, Rhs}.
exercise4() -> fractions_utils:reduce_fraction({84, 126}).
exercise5() -> gcd_lcm:lcm(12, 18).

main() ->
    io:format("Exercise 1 gcd(252,198): ~p~n", [exercise1()]),
    io:format("Exercise 2 lcm(12,18): ~p~n", [exercise2()]),
    {Lhs,Rhs} = exercise3(),
    io:format("Exercise 3 modular add check: ~p ~p~n", [Lhs, Rhs]),
    {Num,Den} = exercise4(),
    io:format("Exercise 4 reduce 84/126: ~p/~p~n", [Num, Den]),
    io:format("Exercise 5 smallest day multiple: ~p~n", [exercise5()]).