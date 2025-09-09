from dec_to_bin import dec_to_bin
from bin_to_dec import bin_to_dec

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg} Expected {expected}, got {actual}")

def test_dec_to_bin():
    assert_eq(dec_to_bin(0), "0")
    assert_eq(dec_to_bin(5), "101")
    assert_eq(dec_to_bin(19), "10011")
    assert_eq(dec_to_bin(42), "101010")

def test_bin_to_dec():
    assert_eq(bin_to_dec("0"), 0)
    assert_eq(bin_to_dec("101"), 5)
    assert_eq(bin_to_dec("10011"), 19)
    assert_eq(bin_to_dec("101010"), 42)

def main():
    test_dec_to_bin()
    test_bin_to_dec()
    print("All tests passed ✔")

if __name__ == "__main__":
    main()
