#ifndef GCD_LCM_H
#define GCD_LCM_H

#include <stdint.h>

int64_t gcd64(int64_t a, int64_t b);
int64_t lcm64(int64_t a, int64_t b); /* lcm(0, b) = 0 by convention */

#endif