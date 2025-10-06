#ifndef INT_ARITH_H
#define INT_ARITH_H

#include <stdint.h>
#include <stdbool.h>

#define INT32_MIN_C  (-2147483647-1)
#define INT32_MAX_C  2147483647
#define INT64_MIN_C  (-9223372036854775807LL-1)
#define INT64_MAX_C  9223372036854775807LL

/* -------- int32 -------- */
bool add_i32_checked(int32_t a, int32_t b, int32_t *out);
int32_t add_i32_wrapping(int32_t a, int32_t b);
int32_t add_i32_saturating(int32_t a, int32_t b);
bool mul_i32_checked(int32_t a, int32_t b, int32_t *out);

/* -------- int64 -------- */
bool add_i64_checked(int64_t a, int64_t b, int64_t *out);
int64_t add_i64_wrapping(int64_t a, int64_t b);
int64_t add_i64_saturating(int64_t a, int64_t b);
bool mul_i64_checked(int64_t a, int64_t b, int64_t *out);

#endif
