#include "int_overflow_sim.h"

uint32_t add_unsigned_bits(uint32_t x, uint32_t y, int bits) {
    uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    return (x + y) & mask;
}

int32_t to_signed_bits(uint32_t value, int bits) {
    uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    uint32_t v = value & mask;
    uint32_t sign = 1u << (bits - 1);
    if (v & sign) {
        /* negative: subtract 2^bits */
        return (int32_t)(v - (sign << 1));
    }
    return (int32_t)v;
}

int32_t add_signed_bits(int32_t x, int32_t y, int bits) {
    uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    uint32_t raw = ((uint32_t)x + (uint32_t)y) & mask;
    return to_signed_bits(raw, bits);
}

void add_with_flags_8bit(uint8_t x, uint8_t y, uint8_t *result, bool *cf, bool *of) {
    uint16_t wide = (uint16_t)x + (uint16_t)y;
    uint8_t r = (uint8_t)(wide & 0xFFu);

    if (result) *result = r;
    if (cf) *cf = (wide > 0xFFu);

    int8_t sx = (int8_t)x;
    int8_t sy = (int8_t)y;
    int8_t sr = (int8_t)r;
    /* Signed overflow: same-sign add yields different-sign result */
    bool overflow = ((sx >= 0 && sy >= 0 && sr < 0) || (sx < 0 && sy < 0 && sr >= 0));
    if (of) *of = overflow;
}
