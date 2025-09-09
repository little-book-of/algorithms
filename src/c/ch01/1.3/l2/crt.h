#ifndef CRT_H
#define CRT_H

#include <stdint.h>
#include <stdbool.h>

/* Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2) for coprime moduli.
   Returns true on success with *x in [0, m1*m2-1], *m = m1*m2. */
bool crt_pair_u64(uint64_t a1, uint64_t m1, uint64_t a2, uint64_t m2, uint64_t *x, uint64_t *m);

#endif