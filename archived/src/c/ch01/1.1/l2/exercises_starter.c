#include <stdio.h>
#include <math.h>
#include <float.h>
#include "ieee754_utils.h"

/* Exercise 1: fields of 1.0 */
static void ex1(void) {
    fp_fields f = decompose_double(1.0);
    printf("Ex1: sign=%d exp_raw=%u mantissa=0x%013llx (unbiased=%d)\n",
           f.sign, (unsigned)f.exponent_raw,
           (unsigned long long)f.mantissa,
           (int)f.exponent_raw - EXP_BIAS);
}

/* Exercise 2: ULP near 1.0 vs DBL_EPSILON */
static void ex2(void) {
    double u = ulp(1.0);
    printf("Ex2: ULP(1.0)=%.17g vs DBL_EPSILON=%.17g\n", u, DBL_EPSILON);
}

/* Exercise 3: classify representative values */
static void ex3(void) {
    double vals[] = {0.0, 1.0, INFINITY, NAN, nextafter(0.0, 1.0)};
    const char *names[] = {"zero", "one", "+inf", "nan", "first_subnormal"};
    for (int i = 0; i < 5; ++i) {
        enum fp_kind k = classify_double(vals[i]);
        const char *kind =
            (k == FP_NAN) ? "nan" :
            (k == FP_INF) ? "inf" :
            (k == FP_ZERO) ? "zero" :
            (k == FP_SUBNORMAL) ? "subnormal" : "normal";
        printf("Ex3: %s -> %s\n", names[i], kind);
    }
}

/* Exercise 4: roundtrip bits for 3.5 */
static void ex4(void) {
    double x = 3.5;
    uint64_t bits = double_to_bits(x);
    double y = bits_to_double(bits);
    printf("Ex4: 3.5 bits=0x%016llx back=%.17g\n",
           (unsigned long long)bits, y);
}

/* Exercise 5: build a +∞ and a NaN explicitly */
static void ex5(void) {
    fp_fields finf = {.sign=0, .exponent_raw=(1u<<EXP_BITS)-1u, .mantissa=0};
    fp_fields fnan = {.sign=0, .exponent_raw=(1u<<EXP_BITS)-1u, .mantissa=1}; /* any nonzero mantissa */
    double inf = build_double(finf);
    double nan = build_double(fnan);
    printf("Ex5: built +inf=%g, built nan==nan? %s\n", inf, (nan==nan) ? "true" : "false");
}

int main(void) {
    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    return 0;
}
