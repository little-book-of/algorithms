-module(exercises_starter).
-export([exercise1/0, exercise2/0, exercise3/0, exercise4/0, exercise5/0, main/0]).

%% 1) modular inverse 7 mod 13
exercise1() -> extgcd_inv:invmod(7, 13).   %% = 2

%% 2) phi(10) and verify Euler’s theorem for a=3
exercise2() ->
    Phi10 = totient:phi(10),  %% 4
    N = 10,
    A = 3 rem N,
    Acc = powmod(A, Phi10, N),
    {Phi10, Acc =:= 1}.

powmod(_Base, 0, _Mod) -> 1;
powmod(Base, Exp, Mod) when (Exp band 1) =:= 1 ->
    (Base * powmod((Base*Base) rem Mod, Exp bsr 1, Mod)) rem Mod;
powmod(Base, Exp, Mod) ->
    powmod((Base*Base) rem Mod, Exp bsr 1, Mod).

%% 3) Fermat on 341 with base 2 -> returns 1 (Carmichael)
exercise3() ->
    powmod(2, 340, 341).

%% 4) Use Miller–Rabin on a small list
exercise4() ->
    [N || N <- [97, 341, 561, 569], miller_rabin:is_probable_prime(N)].

%% 5) CRT: x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7) -> 52 mod 140
exercise5() ->
    {X12, M12} = crt:crt_pair({1,4}, {2,5}),
    crt:crt_pair({X12, M12}, {3,7}).

main() ->
    io:format("Exercise 1 invmod(7,13): ~p~n", [exercise1()]),
    {Phi10, Ok} = exercise2(),
    io:format("Exercise 2 phi(10) & check: ~p ~p~n", [Phi10, Ok]),
    io:format("Exercise 3 Fermat 341 result: ~p~n", [exercise3()]),
    io:format("Exercise 4 probable primes: ~p~n", [exercise4()]),
    {Sol, Mod} = exercise5(),
    io:format("Exercise 5 CRT: ~p mod ~p~n", [Sol, Mod]).