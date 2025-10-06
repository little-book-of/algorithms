#ifndef PARITY_H
#define PARITY_H

#include <stdbool.h>
#include <stdint.h>

bool is_even(int64_t n);
bool is_odd(int64_t n);
/* Return "even" or "odd" using the last binary bit */
const char *parity_bit(int64_t n);

#endif /* PARITY_H */
