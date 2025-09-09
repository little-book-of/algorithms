#include "float_num.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

bool almost_equal(double x, double y, double rel, double abs_) {
    double ax = fabs(x), ay = fabs(y);
    double diff = fabs(x - y);
    double tol = fmax(rel * fmax(ax, ay), abs_);
    return diff <= tol;
}

double kahan_sum(const double *xs, size_t n) {
    double s = 0.0, c = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double y = xs[i] - c;
        double t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

double pairwise_sum(const double *xs, size_t n) {
    if (n == 0) return 0.0;
    double *work = (double *)malloc(n * sizeof(double));
    if (!work) return NAN;
    memcpy(work, xs, n * sizeof(double));
    size_t m = n;
    while (m > 1) {
        size_t k = 0;
        size_t i = 0;
        for (; i + 1 < m; i += 2) {
            work[k++] = work[i] + work[i+1];
        }
        if (i < m) work[k++] = work[i]; /* carry last odd element */
        m = k;
    }
    double out = work[0];
    free(work);
    return out;
}

double machine_epsilon(void) {
    double eps = 1.0;
    while ((1.0 + eps / 2.0) != 1.0) {
        eps /= 2.0;
    }
    return eps;
}

uint64_t ulp_diff(double a, double b) {
    uint64_t ua, ub;
    memcpy(&ua, &a, sizeof ua);
    memcpy(&ub, &b, sizeof ub);
    /* Map negatives to a lexicographically increasing space. For positives,
       raw ordering already matches magnitude. */
    if ((int64_t)ua < 0) ua = 0x8000000000000000ULL - ua;
    if ((int64_t)ub < 0) ub = 0x8000000000000000ULL - ub;
    return (ua > ub) ? (ua - ub) : (ub - ua);
}
