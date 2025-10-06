#include <stdio.h>
#include <inttypes.h>
#include "int_arith.h"
#include "float_num.h"
#include "fixed_point.h"

static void exercise_1_policy(void) {
    int32_t wrap = add_i32_wrapping(INT32_MAX_C, 1);
    int32_t sat  = add_i32_saturating(INT32_MAX_C, 1);
    int32_t chk_val;
    const char *chk = "ok";
    if (!add_i32_checked(INT32_MAX_C, 1, &chk_val)) chk = "OverflowError";
    printf("Exercise 1 (policy): wrap=%d checked=%s sat=%d\n", wrap, chk, sat);
}

static void exercise_2_compare_floats(void) {
    double x = 0.1 + 0.2;
    printf("Exercise 2 (float compare): x=%.17g eq=%s almost=%s\n",
           x, (x == 0.3) ? "true" : "false",
           almost_equal(x, 0.3, 1e-12, 1e-12) ? "true" : "false");
}

static void exercise_3_summation(void) {
    size_t n = 100000;
    double *xs = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) xs[i] = 1.0 / (double)(i + 1);
    double p = pairwise_sum(xs, n), k = kahan_sum(xs, n);
    printf("Exercise 3 (pairwise vs kahan): %.15f %.15f\n", p, k);
    free(xs);
}

static void exercise_4_ledger(void) {
    int64_t a, b, amt;
    parse_dollars_to_cents("12.34", &a);
    parse_dollars_to_cents("0.00", &b);
    parse_dollars_to_cents("0.34", &amt);
    Ledger A, B; ledger_init(&A, a); ledger_init(&B, b);
    ledger_transfer(&A, &B, amt);
    char sa[32], sb[32];
    format_cents(A.balance_cents, sa, sizeof sa);
    format_cents(B.balance_cents, sb, sizeof sb);
    printf("Exercise 4 (ledger): %s %s\n", sa, sb); /* expect 12.00 0.34 */
}

int main(void) {
    exercise_1_policy();
    exercise_2_compare_floats();
    exercise_3_summation();
    exercise_4_ledger();
    return 0;
}
