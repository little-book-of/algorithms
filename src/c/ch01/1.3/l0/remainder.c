#include "remainder.h"
#include <stdbool.h>

void div_identity(int64_t n, int64_t d, int64_t *q, int64_t *r) {
    /* caller must ensure d != 0 */
    *q = n / d;
    *r = n % d;
}

int week_shift(int start, int shift) {
    int m = 7;
    int t = (start % m + m) % m;
    int s = (shift % m + m) % m;
    return (t + s) % m;
}

int64_t powmod(int64_t base, int64_t exp, int64_t mod) {
    if (mod == 1) return 0;
    int64_t res = 1 % mod;
    int64_t b = ((base % mod) + mod) % mod;
    int64_t e = exp;
    while (e > 0) {
        if (e & 1) res = (res * b) % mod;
        b = (b * b) % mod;
        e >>= 1;
    }
    return res;
}
