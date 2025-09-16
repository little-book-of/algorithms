#include "crt.h"
#include "extgcd_inv.h"
#include <inttypes.h>

bool crt_pair_u64(uint64_t a1, uint64_t m1, uint64_t a2, uint64_t m2, uint64_t *x, uint64_t *m) {
    /* Require coprime moduli. */
    /* Simple gcd via extgcd on int64_t (safe for <= 2^63-1); for full u64, a tiny gcd_u64 could be added. */
    int64_t s, t;
    int64_t g = extended_gcd((int64_t)m1, (int64_t)m2, &s, &t);
    if (g != 1 && g != -1) return false;

    /* k = ((a2 - a1) mod m2) * inv(m1 mod m2) mod m2 */
    int64_t inv;
    if (!invmod_i64((int64_t)(m1 % m2), (int64_t)m2, &inv)) return false;
    uint64_t k = ((a2 + m2 - (a1 % m2)) % m2);
    k = ( ( (__uint128_t)k * (uint64_t)inv ) % m2 );

    __uint128_t M = ( (__uint128_t)m1 * (__uint128_t)m2 );
    __uint128_t X = ( (__uint128_t)a1 + (__uint128_t)k * (__uint128_t)m1 ) % M;
    if (x) *x = (uint64_t)X;
    if (m) *m = (uint64_t)M;
    return true;
}