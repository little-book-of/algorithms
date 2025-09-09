#ifndef INT_OVERFLOW_SIM_H
#define INT_OVERFLOW_SIM_H

#include <stdint.h>
#include <stdbool.h>

/* Unsigned addition with wraparound modulo 2^bits. */
uint32_t add_unsigned_bits(uint32_t x, uint32_t y, int bits);

/* Interpret 'value' (0..2^bits-1) as signed two's-complement. */
int32_t to_signed_bits(uint32_t value, int bits);

/* Signed two's-complement addition with wraparound at 'bits'. */
int32_t add_signed_bits(int32_t x, int32_t y, int bits);

/* 8-bit add returning result and flags (CF for unsigned carry, OF for signed overflow). */
void add_with_flags_8bit(uint8_t x, uint8_t y, uint8_t *result, bool *cf, bool *of);

#endif
