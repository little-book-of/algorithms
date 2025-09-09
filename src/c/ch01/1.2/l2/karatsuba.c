#include "karatsuba.h"

long long karatsuba(long long x, long long y) {
    if (x < 10 || y < 10) return x * y;

    int n = (x > y ? x : y);
    int bits = 0;
    while (n > 0) {
        bits++;
        n >>= 1;
    }
    int m = bits / 2;

    long long high1 = x >> m;
    long long low1  = x - (high1 << m);
    long long high2 = y >> m;
    long long low2  = y - (high2 << m);

    long long z0 = karatsuba(low1, low2);
    long long z2 = karatsuba(high1, high2);
    long long z1 = karatsuba(low1 + high1, low2 + high2) - z0 - z2;

    return (z2 << (2 * m)) + (z1 << m) + z0;
}
