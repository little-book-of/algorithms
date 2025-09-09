#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "extgcd_inv.h"
#include "totient.h"
#include "miller_rabin.h"
#include "crt.h"

int64_t exercise_1(void) {
    int64_t inv;
    return invmod_i64(7, 13, &inv) ? inv : -1; /* 2 */
}

void exercise_2(uint64_t *phi10, int *euler_ok) {
    *phi10 = phi_u64(10); /* 4 */
    /* Euler's theorem check for a=3, n=10 */
    *euler_ok = ( ( (uint64_t)1 == 1 ) && ( (uint64_t) ( ( (uint64_t)1 /* placeholder */ ) ) ) );
    *euler_ok = ( ( (uint64_t)1 == (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) ( (uint64_t) 0 ) ) ) ) ) ) ) ) ) ) ); /* will set below */
    /* Just compute powmod via repeated squaring using MR's powmod wrapper indirectly not exposed; re-check via repeated multiply with % 10. */
    uint64_t a = 3, n = 10, e = *phi10, acc = 1 % n, base = a % n;
    while (e) {
        if (e & 1) acc = (acc * base) % n;
        base = (base * base) % n;
        e >>= 1;
    }
    *euler_ok = (acc == 1);
}

uint64_t exercise_3(void) {
    /* Fermat on 341 with base 2: 2^340 mod 341 == 1 (despite composite) */
    uint64_t n = 341, e = 340, acc = 1 % n, base = 2 % n;
    while (e) { if (e & 1) acc = (acc * base) % n; base = (base * base) % n; e >>= 1; }
    return acc;
}

void exercise_4(uint64_t *out, size_t *len) {
    /* filter probable primes from a small list */
    static const uint64_t nums[] = {97, 341, 561, 569};
    size_t k = 0;
    for (unsigned i = 0; i < sizeof(nums)/sizeof(nums[0]); ++i) {
        if (is_probable_prime_u64(nums[i])) out[k++] = nums[i];
    }
    *len = k;
}

void exercise_5(uint64_t *sol, uint64_t *mod) {
    /* x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7) */
    uint64_t x12, m12;
    crt_pair_u64(1, 4, 2, 5, &x12, &m12);
    crt_pair_u64(x12, m12, 3, 7, sol, mod);
}

int main(void) {
    printf("Exercise 1 invmod(7,13): %" PRId64 "\n", exercise_1());

    uint64_t phi10; int ok;
    exercise_2(&phi10, &ok);
    printf("Exercise 2 phi(10) & check: %" PRIu64 " %s\n", phi10, ok ? "true" : "false");

    printf("Exercise 3 Fermat 341 result: %" PRIu64 "\n", exercise_3());

    uint64_t out[8]; size_t len;
    exercise_4(out, &len);
    printf("Exercise 4 probable primes:");
    for (size_t i = 0; i < len; ++i) printf(" %llu", (unsigned long long)out[i]);
    printf("\n");

    uint64_t sol, mod;
    exercise_5(&sol, &mod);
    printf("Exercise 5 CRT: %llu mod %llu\n",
           (unsigned long long)sol, (unsigned long long)mod);
    return 0;
}