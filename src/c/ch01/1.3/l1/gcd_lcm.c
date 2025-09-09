#include "gcd_lcm.h"
#include <stdlib.h> /* llabs */

int64_t gcd64(int64_t a, int64_t b) {
    a = llabs(a);
    b = llabs(b);
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int64_t lcm64(int64_t a, int64_t b) {
    if (a == 0 || b == 0) return 0;
    /* divide before multiply to reduce overflow risk */
    int64_t g = gcd64(a, b);
    return llabs((a / g) * b);
}