#include "totient.h"

uint64_t phi_u64(uint64_t n) {
    if (n == 0) return 0; /* undefined, but keep 0 for safety */
    uint64_t result = n;
    uint64_t x = n;
    uint64_t p = 2;
    while (p * p <= x) {
        if (x % p == 0) {
            while (x % p == 0) x /= p;
            result -= result / p;
        }
        p += (p == 2 ? 1 : 2); /* after 2, check odds */
    }
    if (x > 1) {
        result -= result / x;
    }
    return result;
}