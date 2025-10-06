#include "float_precision.h"
#include <math.h>

bool almost_equal(double x, double y, double eps) {
    return fabs(x - y) < eps;
}

double repeat_add(double value, int times) {
    double s = 0.0;
    for (int i = 0; i < times; ++i) s += value;
    return s;
}

double mix_large_small(double large, double small) {
    return large + small;
}

double sum_naive(const double *xs, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += xs[i];
    return s;
}
