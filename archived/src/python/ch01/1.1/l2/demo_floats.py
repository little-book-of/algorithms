import math
import sys

def main():
    # Rounding surprise
    a = 0.1 + 0.2
    print("0.1 + 0.2 =", a)
    print("Equal to 0.3?", a == 0.3)

    # Special values
    pos_inf = float('inf')
    nan = float('nan')
    print("Infinity:", pos_inf)
    print("NaN == NaN?", nan == nan)  # always False

    # Epsilon & ULP near 1.0
    print("sys.float_info.epsilon:", sys.float_info.epsilon)
    print("math.ulp(1.0):", math.ulp(1.0))

    # Next representable numbers around 1.0 (requires Python 3.9+)
    next_up = math.nextafter(1.0, math.inf)
    next_down = math.nextafter(1.0, -math.inf)
    print("nextafter(1.0, +inf):", next_up)
    print("nextafter(1.0, -inf):", next_down)
    print("ULP around 1.0 (next_up - 1.0):", next_up - 1.0)

    # Underflow / denormal territory example
    tiny = 5e-324  # minimum positive subnormal for IEEE-754 double
    print("Tiny (subnormal) ~5e-324:", tiny)
    print("Is zero if divided by 2?", tiny / 2 == 0.0)

if __name__ == "__main__":
    main()
