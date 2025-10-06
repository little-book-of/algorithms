#ifndef TWOS_COMPLEMENT_H
#define TWOS_COMPLEMENT_H

#include <stdbool.h>
#include <stdint.h>

/* Encode a signed integer x into a two's-complement bitstring of width `bits`.
 * bits must be in [1, 32]. Returns a heap string of length `bits` (no terminator counted).
 * Caller must free().
 * If x is out-of-range for the given width, it will be truncated as if cast to int32_t.
 */
char* to_twos_complement(int32_t x, int bits);

/* Decode a two's-complement bitstring (e.g., "11111001") to a signed int32.
 * The string length is the bit width. Returns true on success.
 * Fails (returns false) if the string contains chars other than '0' or '1', or length==0 or >32.
 */
bool from_twos_complement(const char* s, int32_t* out);

#endif /* TWOS_COMPLEMENT_H */
