#include "bin_to_dec.h"
#include <stddef.h>

bool bin_to_dec(const char* s, unsigned int* out) {
    if (!s || !*s || !out) return false;

    unsigned int value = 0;
    for (const char* p = s; *p; ++p) {
        if (*p != '0' && *p != '1') return false;
        value = (value << 1) | (*p - '0');
    }
    *out = value;
    return true;
}
