#ifndef FLOAT_NUM_H
#define FLOAT_NUM_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* Float compare: |x - y| <= max(rel*max(|x|,|y|), abs) */
bool almost_equal(double x, double y, double rel, double abs);

/* Summation */
double kahan_sum(const double *xs, size_t n);
double pairwise_sum(const double *xs, size_t n);

/* Epsilon & ULP */
double machine_epsilon(void);
uint64_t ulp_diff(double a, double b);

#endif
