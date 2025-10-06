#ifndef FRACTIONS_UTILS_H
#define FRACTIONS_UTILS_H

#include <stdint.h>

typedef struct {
    int64_t num;
    int64_t den;
} frac64;

/* Reduce to lowest terms; canonical sign (denominator positive). d != 0. */
frac64 reduce_fraction64(int64_t n, int64_t d);

#endif