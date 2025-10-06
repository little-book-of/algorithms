#ifndef IEEE754_UTILS_H
#define IEEE754_UTILS_H

#include <stdint.h>
#include <stdbool.h>

enum fp_kind {
    FP_NAN,
    FP_INF,
    FP_ZERO,
    FP_SUBNORMAL,
    FP_NORMAL
};

/* IEEE-754 double (binary64) constants */
enum { MANT_BITS = 52, EXP_BITS = 11, EXP_BIAS = 1023 };

typedef struct {
    int sign;              /* 0 or 1 */
    uint16_t exponent_raw; /* 0..2047 */
    uint64_t mantissa;     /* 52 bits */
} fp_fields;

/* Convert between double and raw 64-bit pattern */
uint64_t double_to_bits(double x);
double   bits_to_double(uint64_t bits);

/* Decompose/build IEEE-754 fields */
fp_fields decompose_double(double x);
double    build_double(fp_fields f);

/* Classify a double into NaN/Inf/Zero/Subnormal/Normal */
enum fp_kind classify_double(double x);

/* Pretty formatting helpers (written to buffer) */
void fields_pretty(double x, char *buf, size_t n);

/* ULP near a given x: distance between x and next representable in +∞ dir */
double ulp(double x);

#endif /* IEEE754_UTILS_H */
