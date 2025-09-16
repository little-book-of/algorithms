#ifndef DEC_TO_BIN_H
#define DEC_TO_BIN_H

#include <stddef.h>

/* Convert an unsigned int to a heap-allocated binary string.
 * Caller must free the returned pointer. Returns NULL on allocation failure.
 * Special case: 0 -> "0".
 */
char* dec_to_bin(unsigned int n);

#endif
