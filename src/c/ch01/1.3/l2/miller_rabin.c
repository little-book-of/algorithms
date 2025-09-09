#include "miller_rabin.h"

/* Safe modular multiply using 128-bit intermediate (GCC/Clang on mainstream 64-bit). */
static inline uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t m) {
    __uint128_t t = ( (__uint128_t)a * (__uint128_t)b ) % m;
    return (uint64_t)t;
}

/* Fast modular exponentiation. */
static uint64_t powmod_u64(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    uint64_t res = 1 % mod;
    uint64_t b = base % mod;
    while (exp > 0) {
        if (exp & 1) res = mulmod_u64(res, b, mod);
        b = mulmod_u64(b, b, mod);
        exp >>= 1;
    }
    return res;
}

static inline void decompose(uint64_t n_minus_1, uint64_t *d, uint32_t *s) {
    uint64_t dd = n_minus_1;
    uint32_t ss = 0;
    while ((dd & 1ULL) == 0) { dd >>= 1; ++ss; }
    *d = dd; *s = ss;
}

static bool try_composite(uint64_t a, uint64_t d, uint64_t n, uint32_t s) {
    uint64_t x = powmod_u64(a, d, n);
    if (x == 1 || x == n - 1) return false;
    for (uint32_t i = 1; i < s; ++i) {
        x = mulmod_u64(x, x, n);
        if (x == n - 1) return false;
    }
    return true; /* definitely composite for this base */
}

bool is_probable_prime_u64(uint64_t n) {
    if (n < 2) return false;
    /* small primes & quick divisibility */
    const uint32_t small_primes[] = {2,3,5,7,11,13,17,19,23,29};
    for (unsigned i = 0; i < sizeof(small_primes)/sizeof(small_primes[0]); ++i) {
        uint32_t p = small_primes[i];
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    uint64_t d; uint32_t s;
    decompose(n - 1, &d, &s);

    /* Deterministic base set for 64-bit range. */
    const uint64_t bases[] = {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL};
    for (unsigned i = 0; i < sizeof(bases)/sizeof(bases[0]); ++i) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;
        if (try_composite(a, d, n, s)) return false;
    }
    return true;
}