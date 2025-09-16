#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "twos_complement.h"

static void ex1(void) {
    /* Convert decimal 100 to octal and hex via printf formats. */
    unsigned int n = 100;
    printf("Exercise 1 (100): oct=%o hex=%X\n", n, n);
}

static void ex2(void) {
    /* Write -7 in 8-bit two's complement. */
    char* s = to_twos_complement(-7, 8);
    if (!s) { puts("alloc failed"); return; }
    printf("Exercise 2 (-7 in 8-bit): %s\n", s);  /* expect 11111001 */
    free(s);
}

static void ex3(void) {
    /* Verify 0xFF == 255 */
    printf("Exercise 3 (0xFF == 255?): %s\n", (0xFFu == 255u) ? "true" : "false");
}

static void ex4(void) {
    /* Parse "11111001" as 8-bit two's complement. */
    int32_t val;
    if (from_twos_complement("11111001", &val)) {
        printf("Exercise 4 (parse '11111001'): %d\n", val); /* expect -7 */
    } else {
        puts("Exercise 4: invalid input");
    }
}

int main(void) {
    ex1();
    ex2();
    ex3();
    ex4();
    return 0;
}
