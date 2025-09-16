#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "int_overflow_sim.h"
#include "float_precision.h"

int main(void) {
    puts("=== Integer overflow (8-bit) ===");
    printf("255 + 1 (unsigned, wrap) -> %u\n",
           add_unsigned_bits(255u, 1u, 8));                 /* 0 */
    printf("127 + 1 (signed, wrap)   -> %d\n",
           add_signed_bits(127, 1, 8));                     /* -128 */

    uint8_t r; bool cf, of;
    add_with_flags_8bit(200u, 100u, &r, &cf, &of);
    printf("200 + 100 -> result=%u (unsigned), CF=%s, OF=%s\n",
           (unsigned)r, cf ? "true" : "false", of ? "true" : "false");
    printf("Interpret result as signed: %d\n", to_signed_bits(r, 8));

    puts("\n=== Floating-point surprises ===");
    double x = 0.1 + 0.2;
    printf("0.1 + 0.2 = %.17g\n", x);                     /* 0.30000000000000004 */
    printf("Direct equality with 0.3? %s\n", (x == 0.3) ? "true" : "false");
    printf("Using epsilon: %s\n", almost_equal(x, 0.3, 1e-9) ? "true" : "false");

    puts("\nRepeat add 0.1 ten times:");
    double s = repeat_add(0.1, 10);
    printf("sum = %.17g, equal to 1.0? %s, almost_equal? %s\n",
           s, (s == 1.0) ? "true" : "false",
           almost_equal(s, 1.0, 1e-9) ? "true" : "false");

    puts("\nMix large and small:");
    double a = mix_large_small(1e16, 1.0);
    printf("1e16 + 1.0 = %.17g (often unchanged)\n", a);

    puts("\nNaive sum may drift slightly:");
    double nums[10];
    for (int i = 0; i < 10; ++i) nums[i] = 0.1;
    printf("sum_naive([0.1]*10) = %.17g\n", sum_naive(nums, 10));

    return 0;
}
