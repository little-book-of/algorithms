#ifndef EXTGCD_INV_H
#define EXTGCD_INV_H

#include <stdint.h>
#include <stdbool.h>

/* Extended GCD: returns g = gcd(a,b) and Bézout x,y such that a*x + b*y = g */
int64_t extended_gcd(int64_t a, int64_t b, int64_t *x, int64_t *y);

/* Modular inverse: find inv in [0,m-1] with (a*inv) % m == 1; returns false if gcd(a,m) != 1. */
bool invmod_i64(int64_t a, int64_t m, int64_t *inv);

#endif