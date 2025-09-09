from int_overflow_sim import add_unsigned, add_signed, add_with_flags_8bit, to_signed
from float_precision import almost_equal, repeat_add, mix_large_small, sum_list

def main():
    print("=== Integer overflow (8-bit) ===")
    print("255 + 1 (unsigned, wrap):", add_unsigned(255, 1, bits=8))  # 0
    print("127 + 1 (signed, wrap):  ", add_signed(127, 1, bits=8))    # -128

    r, cf, of = add_with_flags_8bit(200, 100)
    print(f"200 + 100 -> result={r} (unsigned), CF={cf}, OF={of}")
    print("Interpret result as signed:", to_signed(r, 8))

    print("\n=== Floating-point surprises ===")
    x = 0.1 + 0.2
    print("0.1 + 0.2 =", x)                  # 0.30000000000000004
    print("Direct equality with 0.3?", x == 0.3)
    print("Using epsilon:", almost_equal(x, 0.3))  # True

    print("\nRepeat add 0.1 ten times:")
    s = repeat_add(0.1, 10)
    print("sum =", s, "equal to 1.0?", s == 1.0, "almost_equal?", almost_equal(s, 1.0))

    print("\nMix large and small:")
    a = mix_large_small(1e16, 1.0)
    print("1e16 + 1.0 =", a, " (unchanged)")

    print("\nNaive sum may drift slightly:")
    nums = [0.1] * 10
    print("sum_list([0.1]*10) =", sum_list(nums))

if __name__ == "__main__":
    main()
