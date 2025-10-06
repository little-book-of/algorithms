#include "fractions_utils.h"
#include "gcd_lcm.h"

frac64 reduce_fraction64(int64_t n, int64_t d) {
    /* caller must ensure d != 0 */
    int64_t g = gcd64(n, d);
    int64_t num = n / g;
    int64_t den = d / g;
    if (den < 0) { num = -num; den = -den; }
    return (frac64){ .num = num, .den = den };
}