#ifndef TOTIENT_H
#define TOTIENT_H

#include <stdint.h>

/* Euler's totient φ(n), for n >= 1; teaching version with trial factorization. */
uint64_t phi_u64(uint64_t n);

#endif