#include "dec_to_bin.h"
#include <stdlib.h>
#include <string.h>

char* dec_to_bin(unsigned int n) {
    if (n == 0) {
        char* s = malloc(2);
        if (!s) return NULL;
        s[0] = '0'; s[1] = '\0';
        return s;
    }

    char buf[sizeof(unsigned int) * 8 + 1]; // enough bits
    size_t i = 0;

    while (n > 0) {
        buf[i++] = (char)('0' + (n % 2));
        n /= 2;
    }
    buf[i] = '\0';

    char* out = malloc(i + 1);
    if (!out) return NULL;

    for (size_t j = 0; j < i; j++) {
        out[j] = buf[i - 1 - j];
    }
    out[i] = '\0';
    return out;
}
