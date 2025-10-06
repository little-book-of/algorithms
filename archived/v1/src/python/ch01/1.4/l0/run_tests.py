from int_overflow_sim import add_unsigned, add_signed, to_signed
from float_precision import almost_equal, repeat_add

def assert_eq(got, expected, msg=""):
    if got != expected:
        raise AssertionError(f"{msg} expected {expected}, got {got}")

def test_unsigned_wrap():
    assert_eq(add_unsigned(255, 1, 8), 0)
    assert_eq(add_unsigned(15, 1, 4), 0)
    assert_eq(add_unsigned(65535, 1, 16), 0)

def test_signed_wrap():
    assert_eq(add_signed(127, 1, 8), -128)
    assert_eq(add_signed(-1, -1, 8), -2)

def test_to_signed():
    assert_eq(to_signed(0x80, 8), -128)
    assert_eq(to_signed(0x7F, 8), 127)

def test_float_basics():
    s = repeat_add(0.1, 10)
    # Not guaranteed to be exactly 1.0, but should be very close.
    assert almost_equal(s, 1.0)

def main():
    test_unsigned_wrap()
    test_signed_wrap()
    test_to_signed()
    test_float_basics()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
