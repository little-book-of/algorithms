#ifndef REMAINDER_H
#define REMAINDER_H

#include <stdint.h>

/* Return quotient and remainder via pointers (matches Python's divmod). */
void div_identity(int64_t n, int64_t d, int64_t *q, int64_t *r);

/* 0..6 week wrap (0=Mon). */
int week_shift(int start, int shift);

/* Fast modular exponent: (base^exp) % mod, using binary exponentiation. */
int64_t powmod(int64_t base, int64_t exp, int64_t mod);

#endif /* REMAINDER_H */
