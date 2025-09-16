%% Fixed-point cents parsing/formatting and a tiny ledger.
-module(fixed_point).
-export([
    parse_dollars_to_cents/1, format_cents/1,
    ledger_new/1, ledger_deposit/2, ledger_withdraw/2, ledger_transfer/3
]).

-record(ledger, {balance_cents = 0 :: integer()}).

%% Parse "123", "123.4", "123.45", "-0.07" -> {ok, Cents} or {error, Reason}
parse_dollars_to_cents(Str0) when is_list(Str0); is_binary(Str0) ->
    S1 = unicode:characters_to_list(Str0),
    S = string:trim(S1),
    case S of
        [] -> {error, empty};
        _  -> parse_money_(S)
    end.

parse_money_([SignCh|Rest]) when SignCh =:= $+; SignCh =:= $- ->
    {Sign, S} = case SignCh of $- -> {-1, Rest}; _ -> {1, Rest} end,
    parse_money_core(Sign, S);
parse_money_(S) ->
    parse_money_core(1, S).

parse_money_core(Sign, S) ->
    {WholeDigits, Tail1} = take_digits(S, []),
    WholeStr = case WholeDigits of [] -> "0"; _ -> lists:reverse(WholeDigits) end,
    case Tail1 of
        [] ->
            with_result(Sign, WholeStr, "00");
        [$.|T2] ->
            {F1, T3} = take_digits(T2, []),
            Fr = lists:reverse(F1),
            Fr2 = pad_trunc_2(Fr),
            case string:trim(T3) of
                [] -> with_result(Sign, WholeStr, Fr2);
                _  -> {error, trailing}
            end;
        _Other ->
            {error, badliteral}
    end.

take_digits([Ch|T], Acc) when Ch >= $0, Ch =< $9 -> take_digits(T, [Ch|Acc]);
take_digits(T, Acc) -> {Acc, T}.

pad_trunc_2(Fr) ->
    case Fr of
        []      -> "00";
        [A]     -> [A,$0];
        [A,B|_] -> [A,B]
    end.

with_result(Sign, WholeStr, Fr2) ->
    case (all_digits(WholeStr) andalso all_digits(Fr2)) of
        false -> {error, badliteral};
        true  ->
            Whole = list_to_integer(WholeStr),
            Cents0 = Whole * 100 + list_to_integer(Fr2),
            {ok, Sign * Cents0}
    end.

all_digits([]) -> true;
all_digits([Ch|T]) when Ch >= $0, Ch =< $9 -> all_digits(T);
all_digits(_) -> false.

format_cents(Cents) ->
    Sign = if Cents < 0 -> "-"; true -> "" end,
    Abs = abs(Cents),
    Dollars = Abs div 100,
    Fr = Abs rem 100,
    lists:flatten(io_lib:format("~s~B.~2..0B", [Sign, Dollars, Fr])).

ledger_new(Cents) ->
    #ledger{balance_cents = Cents}.

ledger_deposit(#ledger{balance_cents = B}=L, Cents) ->
    L#ledger{balance_cents = B + Cents}.

ledger_withdraw(#ledger{balance_cents = B}=L, Cents) ->
    case Cents =< B of
        true  -> {ok, L#ledger{balance_cents = B - Cents}};
        false -> {error, insufficient_funds}
    end.

ledger_transfer(From=#ledger{}, To=#ledger{}, Cents) ->
    case ledger_withdraw(From, Cents) of
        {error, _}=E -> {E, From, To};
        {ok, From1}  ->
            To1 = ledger_deposit(To, Cents),
            {ok, From1, To1}
    end.
