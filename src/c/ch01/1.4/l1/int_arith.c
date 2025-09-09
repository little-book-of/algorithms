#include "int_arith.h"

/* ---------- int32 ---------- */

bool add_i32_checked(int32_t a, int32_t b, int32_t *out) {
    if ((b > 0 && a > INT32_MAX_C - b) ||
        (b < 0 && a < INT32_MIN_C - b)) {
        return false;
    }
    if (out) *out = (int32_t)(a + b);
    return true;
}

int32_t add_i32_wrapping(int32_t a, int32_t b) {
    uint32_t ua = (uint32_t)a, ub = (uint32_t)b;
    return (int32_t)(ua + ub); /* two's complement wrap */
}

int32_t add_i32_saturating(int32_t a, int32_t b) {
    if ((b > 0 && a > INT32_MAX_C - b)) return INT32_MAX_C;
    if ((b < 0 && a < INT32_MIN_C - b)) return INT32_MIN_C;
    return (int32_t)(a + b);
}

bool mul_i32_checked(int32_t a, int32_t b, int32_t *out) {
    /* widen to int64 for safe check */
    int64_t p = (int64_t)a * (int64_t)b;
    if (p > INT32_MAX_C || p < INT32_MIN_C) return false;
    if (out) *out = (int32_t)p;
    return true;
}

/* ---------- int64 ---------- */

bool add_i64_checked(int64_t a, int64_t b, int64_t *out) {
    if ((b > 0 && a > INT64_MAX_C - b) ||
        (b < 0 && a < INT64_MIN_C - b)) {
        return false;
    }
    if (out) *out = a + b;
    return true;
}

int64_t add_i64_wrapping(int64_t a, int64_t b) {
    uint64_t ua = (uint64_t)a, ub = (uint64_t)b;
    return (int64_t)(ua + ub); /* two's complement wrap */
}

int64_t add_i64_saturating(int64_t a, int64_t b) {
    if ((b > 0 && a > INT64_MAX_C - b)) return INT64_MAX_C;
    if ((b < 0 && a < INT64_MIN_C - b)) return INT64_MIN_C;
    return a + b;
}

bool mul_i64_checked(int64_t a, int64_t b, int64_t *out) {
    /* Portable overflow test via division bounds (no __int128 needed). */
    if (a == 0 || b == 0) { if (out) *out = 0; return true; }
    if (a > 0) {
        if (b > 0) { if (a > INT64_MAX_C / b) return false; }
        else       { if (b < INT64_MIN_C / a) return false; }
    } else {
        if (b > 0) { if (a < INT64_MIN_C / b) return false; }
        else       { if (a != 0 && b < INT64_MAX_C / a) return false; }
    }
    if (out) *out = a * b;
    return true;
}
