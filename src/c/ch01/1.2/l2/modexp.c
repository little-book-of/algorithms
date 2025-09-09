#include "modexp.h"

long long modexp(long long a, long long b, long long m) {
    long long result = 1;
    long long base = a % m;
    long long exp = b;

    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    return result;
}
