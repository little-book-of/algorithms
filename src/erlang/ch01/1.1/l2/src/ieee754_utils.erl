%% IEEE-754 (binary64) helpers: decompose/build, classify, ULP, nextafter(+inf).
-module(ieee754_utils).

-export([
    float_to_bits/1, bits_to_float/1,
    decompose/1, build/3,
    classify/1, components_pretty/1,
    unbiased_exp/1, ulp/1,
    nextafter_pos_inf/1
]).

-define(MANT_BITS, 52).
-define(EXP_BITS, 11).
-define(EXP_BIAS, 1023).

%% --- bit <-> float ---

float_to_bits(X) when is_float(X) ->
    <<Bits:64/unsigned-integer>> = <<X:64/float>>,
    Bits.

bits_to_float(Bits) when is_integer(Bits) ->
    <<F:64/float>> = <<Bits:64/unsigned-integer>>,
    F.

%% --- fields ---

%% decompose/1 -> {Sign, ExpRaw, Mantissa}
decompose(X) when is_float(X) ->
    B = float_to_bits(X),
    Sign    = (B bsr 63) band 16#1,
    ExpRaw  = (B bsr ?MANT_BITS) band ((1 bsl ?EXP_BITS) - 1),
    Mant    = B band ((1 bsl ?MANT_BITS) - 1),
    {Sign, ExpRaw, Mant}.

%% build/3 from fields (Sign 0|1, ExpRaw 0..2047, Mant 52 bits)
build(Sign, ExpRaw, Mant) ->
    Bits = ((Sign band 1) bsl 63)
         bor ((ExpRaw band ((1 bsl ?EXP_BITS)-1)) bsl ?MANT_BITS)
         bor (Mant band ((1 bsl ?MANT_BITS)-1)),
    bits_to_float(Bits).

unbiased_exp(ExpRaw) -> ExpRaw - ?EXP_BIAS.

%% classify -> one of: nan | inf | zero | subnormal | normal
classify(X) ->
    {_, E, M} = decompose(X),
    case E of
        ER when ER =:= ((1 bsl ?EXP_BITS) - 1) ->
            case M of 0 -> inf; _ -> nan end;
        0 ->
            case M of 0 -> zero; _ -> subnormal end;
        _ -> normal
    end.

components_pretty(X) ->
    {S,E,M} = decompose(X),
    Kind = classify(X),
    io_lib:format("x=~p kind=~s sign=~B exponent_raw=~B mantissa=0x~13.16.0B",
                  [X, atom_to_list(Kind), S, E, M]).

%% ULP at X: spacing to next representable toward +inf.
%% For normal numbers: 2^(unbiased_exp - 52). For zero/subnormal: 2^-1074.
%% For inf/nan: returns 'nan'.
ulp(X) ->
    case classify(X) of
        normal ->
            {_,E,_} = decompose(X),
            math:pow(2, unbiased_exp(E) - ?MANT_BITS);
        zero ->
            math:pow(2, -1074);
        subnormal ->
            math:pow(2, -1074);
        _ ->
            'nan'
    end.

%% nextafter toward +infinity implemented by bit tweak:
%% For +0.0, return smallest subnormal. For +x: increment bits. For -x: decrement bits.
%% For inf/nan: returns X (inf) or NaN as-is.
nextafter_pos_inf(X) when is_float(X) ->
    case classify(X) of
        nan -> X;
        inf -> X;
        zero ->
            %% smallest positive subnormal = 0x0000...0001
            bits_to_float(1);
        _ ->
            Bits0 = float_to_bits(X),
            %% For negative numbers, increasing value toward +inf means decrementing bit pattern.
            Sign = (Bits0 bsr 63) band 1,
            Bits1 = case Sign of
                        0 -> Bits0 + 1;
                        1 -> Bits0 - 1
                    end,
            bits_to_float(Bits1)
    end.
