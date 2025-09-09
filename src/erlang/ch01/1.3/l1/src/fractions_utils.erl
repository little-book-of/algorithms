-module(fractions_utils).
-export([reduce_fraction/1]).

%% reduce_fraction({N,D}) -> {Num, Den} in lowest terms with Den > 0.
reduce_fraction({_N, 0}) ->
    erlang:error(badarg);
reduce_fraction({N, D}) ->
    G = gcd_lcm:gcd(N, D),
    Num0 = N div G,
    Den0 = D div G,
    case Den0 < 0 of
        true  -> {-Num0, -Den0};
        false -> {Num0, Den0}
    end.