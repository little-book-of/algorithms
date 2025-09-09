#ifndef MILLER_RABIN_H
#define MILLER_RABIN_H

#include <stdint.h>
#include <stdbool.h>

/* Deterministic Miller–Rabin for 64-bit range using a fixed base set. */
bool is_probable_prime_u64(uint64_t n);

#endif