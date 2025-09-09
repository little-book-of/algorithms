#include <stdio.h>
#include <stdint.h>
#include "gcd_lcm.h"
#include "modular_identities.h"
#include "fractions_utils.h"

int main(void) {
    /* GCD / LCM */
    int64_t a = 252, b = 198;
    printf("gcd(%lld, %lld) = %lld\n", (long long)a, (long long)b, (long long)gcd64(a,b));
    printf("lcm(12, 18) = %lld\n", (long long)lcm64(12, 18));

    /* Modular identities */
    int64_t x = 123, y = 456, m = 7;
    printf("(x+y)%%m vs mod_add: %lld %lld\n",
           (long long)((x + y) % m), (long long)mod_add64(x, y, m));
    printf("(x*y)%%m vs mod_mul: %lld %lld\n",
           (long long)((x * y) % m), (long long)mod_mul64(x, y, m));

    /* Fraction reduction */
    frac64 r = reduce_fraction64(84, 126);
    printf("reduce_fraction(84,126) = %lld/%lld\n", (long long)r.num, (long long)r.den);

    return 0;
}