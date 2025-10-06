#include <stdio.h>
#include <stdint.h>

int main(void) {
    unsigned int n = 42;

    /* Print same value in decimal, octal, hex */
    printf("Decimal: %u\n", n);
    printf("Octal  : %o\n", n);  /* base 8 */
    printf("Hex    : %X\n", n);  /* base 16 (uppercase) */

    /* Show literal forms: C supports octal (0 prefix) and hex (0x). */
    unsigned int oct_lit = 052;   /* 42 in octal */
    unsigned int hex_lit = 0x2A;  /* 42 in hex */
    printf("Octal literal 052 -> %u\n", oct_lit);
    printf("Hex literal 0x2A -> %u\n", hex_lit);

    return 0;
}
