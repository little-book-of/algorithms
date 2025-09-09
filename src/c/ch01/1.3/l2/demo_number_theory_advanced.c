#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include "extgcd_inv.h"
#include "totient.h"
#include "miller_rabin.h"
#include "crt.h"

int main(void) {
    /* Extended Euclid / inverse */
    int64_t x, y;
    int64_t g = extended_gcd(240, 46, &x, &y);
    printf("extended_gcd(240,46) -> (g=%" PRId64 ", x=%" PRId64 ", y=%" PRId64 ")\n", g, x, y);

    int64_t inv;
    if (invmod_i64(3, 7, &inv))
        printf("invmod(3,7) = %" PRId64 "\n", inv);
    else
        printf("invmod(3,7) failed\n");

    /* Totient */
    printf("phi(10) = %" PRIu64 "\n", phi_u64(10));
    printf("phi(36) = %" PRIu64 "\n", phi_u64(36));

    /* Miller–Rabin */
    uint64_t candidates[] = {97, 561, 1105, 2147483647ULL};
    for (unsigned i = 0; i < sizeof(candidates)/sizeof(candidates[0]); ++i) {
        uint64_t n = candidates[i];
        printf("is_probable_prime(%llu) = %s\n",
               (unsigned long long)n, is_probable_prime_u64(n) ? "true" : "false");
    }

    /* CRT: x≡2 (mod 3), x≡3 (mod 5), x≡2 (mod 7) -> 23 mod 105 */
    uint64_t a = 2, m = 3, a2 = 3, m2 = 5, x12, m12;
    crt_pair_u64(a, m, a2, m2, &x12, &m12);
    uint64_t res, mod;
    crt_pair_u64(x12, m12, 2, 7, &res, &mod);
    printf("CRT -> %llu mod %llu\n", (unsigned long long)res, (unsigned long long)mod);

    return 0;
}