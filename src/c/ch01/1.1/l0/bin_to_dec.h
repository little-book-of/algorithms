#ifndef BIN_TO_DEC_H
#define BIN_TO_DEC_H

#include <stdbool.h>

/* Parse a binary string into unsigned int.
 * Returns true on success, false on invalid input.
 */
bool bin_to_dec(const char* s, unsigned int* out);

#endif
