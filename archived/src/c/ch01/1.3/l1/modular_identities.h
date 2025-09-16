#ifndef MODULAR_IDENTITIES_H
#define MODULAR_IDENTITIES_H

#include <stdint.h>

int64_t mod_norm(int64_t x, int64_t m);              /* Normalize to [0, m-1] (m>0) */
int64_t mod_add64(int64_t a, int64_t b, int64_t m);  /* (a + b) % m */
int64_t mod_sub64(int64_t a, int64_t b, int64_t m);  /* (a - b) % m */
int64_t mod_mul64(int64_t a, int64_t b, int64_t m);  /* (a * b) % m */

#endif