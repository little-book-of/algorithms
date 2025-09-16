#include "twos_complement.h"
#include <stdlib.h>
#include <string.h>

char* to_twos_complement(int32_t x, int bits) {
    if (bits < 1 || bits > 32) return NULL;

    /* Represent x as unsigned with truncation to `bits`. */
    uint32_t mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    uint32_t ux = ((uint32_t)x) & mask;

    char* out = malloc((size_t)bits + 1);
    if (!out) return NULL;

    for (int i = bits - 1; i >= 0; --i) {
        out[i] = (ux & 1u) ? '1' : '0';
        ux >>= 1;
    }
    out[bits] = '\0';
    return out;
}

bool from_twos_complement(const char* s, int32_t* out) {
    if (!s || !*s || !out) return false;

    size_t len = strlen(s);
    if (len == 0 || len > 32) return false;

    uint32_t val = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = s[i];
        if (c == '0' || c == '1') {
            val = (val << 1) | (uint32_t)(c - '0');
        } else {
            return false;
        }
    }

    /* Interpret as signed two's-complement with width=len. */
    if (s[0] == '1') {
        /* Negative number: subtract 2^len */
        uint32_t top = (len == 32) ? 0u : (1u << len);
        int64_t signed_val = (int64_t)val - (int64_t)((len == 32) ? (1ull << 32) : top);
        *out = (int32_t)signed_val;
    } else {
        *out = (int32_t)val;
    }
    return true;
}
