#ifndef FLOAT_PRECISION_H
#define FLOAT_PRECISION_H

#include <stddef.h>
#include <stdbool.h>

/* Return true if |x - y| < eps (simple beginner-safe comparison). */
bool almost_equal(double x, double y, double eps);

/* Add 'value' to 0.0 'times' times (demonstrates small drift, e.g., value=0.1). */
double repeat_add(double value, int times);

/* Adding tiny to huge often makes no difference (limited precision). */
double mix_large_small(double large, double small);

/* Naive left-to-right sum of an array (demonstrates accumulated error). */
double sum_naive(const double *xs, size_t n);

#endif
