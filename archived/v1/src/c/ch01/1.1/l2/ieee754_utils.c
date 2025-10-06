#include "ieee754_utils.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

uint64_t double_to_bits(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

double bits_to_double(uint64_t bits) {
    double x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

fp_fields decompose_double(double x) {
    uint64_t b = double_to_bits(x);
    fp_fields f;
    f.sign = (int)((b >> 63) & 0x1u);
    f.exponent_raw = (uint16_t)((b >> MANT_BITS) & ((1u << EXP_BITS) - 1u));
    f.mantissa = b & ((1ull << MANT_BITS) - 1ull);
    return f;
}

double build_double(fp_fields f) {
    uint64_t sign = ((uint64_t)(f.sign & 1)) << 63;
    uint64_t exp  = ((uint64_t)(f.exponent_raw & ((1u << EXP_BITS) - 1u))) << MANT_BITS;
    uint64_t man  = (f.mantissa & ((1ull << MANT_BITS) - 1ull));
    return bits_to_double(sign | exp | man);
}

enum fp_kind classify_double(double x) {
    fp_fields f = decompose_double(x);
    if (f.exponent_raw == (1u << EXP_BITS) - 1u) {
        return (f.mantissa != 0) ? FP_NAN : FP_INF;
    }
    if (f.exponent_raw == 0) {
        return (f.mantissa == 0) ? FP_ZERO : FP_SUBNORMAL;
    }
    return FP_NORMAL;
}

void fields_pretty(double x, char *buf, size_t n) {
    fp_fields f = decompose_double(x);
    enum fp_kind k = classify_double(x);
    const char *kind =
        (k == FP_NAN) ? "nan" :
        (k == FP_INF) ? "inf" :
        (k == FP_ZERO) ? "zero" :
        (k == FP_SUBNORMAL) ? "subnormal" : "normal";
    snprintf(buf, n,
             "x=%g kind=%s sign=%d exponent_raw=%u mantissa=0x%013llx",
             x, kind, f.sign, (unsigned)f.exponent_raw,
             (unsigned long long)f.mantissa);
}

double ulp(double x) {
    /* Next representable toward +inf */
    double next = nextafter(x, INFINITY);
    return fabs(next - x);
}
