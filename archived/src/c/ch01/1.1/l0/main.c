#include <stdio.h>
#include <stdlib.h>
#include "dec_to_bin.h"
#include "bin_to_dec.h"

int main(void) {
    unsigned int n = 42;
    char* s = dec_to_bin(n);
    if (!s) {
        fputs("Allocation failed\n", stderr);
        return 1;
    }
    printf("Decimal: %u\n", n);
    printf("Binary : %s\n", s);
    free(s);

    const char* bin_str = "1011";
    unsigned int val;
    if (bin_to_dec(bin_str, &val)) {
        printf("Binary string '%s' -> decimal: %u\n", bin_str, val);
    } else {
        printf("Invalid binary string: '%s'\n", bin_str);
    }

    return 0;
}
