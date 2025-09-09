%% Float utilities: almost_equal, Kahan & pairwise sum, machine epsilon, ULP distance, nextafter_up.
-module(float_num).
-export([almost_equal/3, kahan_sum/1, pairwise_sum/1, machine_epsilon/0, ulp_diff/2, nextafter_up/1]).

%% Robust float comparison: |X-Y| =< max(Rel*max(|X|,|Y|), Abs)
almost_equal(X, Y, Rel, Abs) ->
    Diff = abs(X - Y),
    Ax = abs(X), Ay = abs(Y),
    Tol = max(Rel * max(Ax, Ay), Abs),
    Diff =< Tol.

kahan_sum(List) ->
    kahan_sum_(List, 0.0, 0.0).
kahan_sum_([], S, _C) -> S;
kahan_sum_([X|Xs], S, C) ->
    Y = X - C,
    T = S + Y,
    C1 = (T - S) - Y,
    kahan_sum_(Xs, T, C1).

pairwise_sum([]) -> 0.0;
pairwise_sum([X]) -> X;
pairwise_sum(List) ->
    pairwise_sum(pair_once(List, [])).

pair_once([], Acc) -> lists:reverse(Acc);
pair_once([A], Acc) -> lists:reverse([A|Acc]);
pair_once([A,B|Rest], Acc) -> pair_once(Rest, [A+B|Acc]).

machine_epsilon() ->
    machine_epsilon_(1.0).
machine_epsilon_(Eps) ->
    case 1.0 + Eps / 2.0 of
        1.0 -> Eps;
        _   -> machine_epsilon_(Eps / 2.0)
    end.

%% ULP distance between two floats (64-bit IEEE754).
ulp_diff(A, B) ->
    UA = float_bits_u64(A),
    UB = float_bits_u64(B),
    MA = map_for_order(UA),
    MB = map_for_order(UB),
    abs(MA - MB).

%% Next representable float strictly greater than F.
nextafter_up(F) when is_float(F) ->
    Bits = float_bits_u64(F),
    SignBit = Bits band 16#8000000000000000,
    NextBits = case SignBit of
        0 -> Bits + 1;                      %% positive or +0
        _ -> Bits - 1                       %% negative or -0
    end,
    bits_to_float(NextBits).

%% --- bit helpers ---

float_bits_u64(F) ->
    %% big-endian canonical form
    <<Bits:64/unsigned-big-integer>> = <<F:64/float-big>>,
    Bits.

bits_to_float(Bits) ->
    <<F:64/float-big>> = <<Bits:64/unsigned-big-integer>>,
    F.

map_for_order(UnsignedBits) ->
    %% Convert to a lexicographically increasing integer space for ULP diff
    Sign = UnsignedBits band 16#8000000000000000,
    case Sign of
        0 -> UnsignedBits;
        _ -> 16#8000000000000000 - UnsignedBits
    end.

max(A,B) -> if A >= B -> A; true -> B end.
