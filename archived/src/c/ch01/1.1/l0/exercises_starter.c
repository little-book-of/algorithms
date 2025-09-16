#include <stdio.h>
#include <stdlib.h>
#include "dec_to_bin.h"
#include "bin_to_dec.h"

static void ex1(void) {
    char* s = dec_to_bin(19);
    printf("Exercise 1 (19 -> binary): %s\n", s);
    free(s);
}

static void ex2(void) {
    unsigned int out;
    if (bin_to_dec("10101", &out)) {
        printf("Exercise 2 ('10101' -> dec): %u\n", out);
    }
}

static void ex3(void) {
    char* s = dec_to_bin(27);
    printf("Exercise 3 (27 -> binary): %s\n", s);
    free(s);
}

static void ex4(void) {
    unsigned int out;
    if (bin_to_dec("111111", &out)) {
        printf("Exercise 4: 111111 -> %u == 63? %s\n", out,
               out == 63 ? "true" : "false");
    }
}

static void ex5(void) {
    puts("Exercise 5: Binary fits hardware with two stable states (0/1).");
}

int main(void) {
    ex1();
    ex2();
    ex3();
    ex4();
    ex5();
    return 0;
}
