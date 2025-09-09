#include "extgcd_inv.h"
#include <stdlib.h>

int64_t extended_gcd(int64_t a, int64_t b, int64_t *x, int64_t *y) {
    int64_t old_r = a, r = b;
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    while (r != 0) {
        int64_t q = old_r / r;
        int64_t tmp;

        tmp = old_r - q * r; old_r = r; r = tmp;
        tmp = old_s - q * s; old_s = s; s = tmp;
        tmp = old_t - q * t; old_t = t; t = tmp;
    }
    if (x) *x = old_s;
    if (y) *y = old_t;
    return old_r; /* gcd */
}

bool invmod_i64(int64_t a, int64_t m, int64_t *inv) {
    if (m <= 0) return false;
    int64_t x, y;
    int64_t g = extended_gcd(a, m, &x, &y);
    if (g != 1 && g != -1) return false;
    /* Normalize into [0, m-1] */
    int64_t res = x % m;
    if (res < 0) res += m;
    if (inv) *inv = res;
    return true;
}