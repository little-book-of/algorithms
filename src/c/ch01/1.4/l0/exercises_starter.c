#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "int_overflow_sim.h"
#include "float_precision.h"

/* Exercise 1: 4-bit counter sequence starting at 14, add 1 three times. */
static void exercise_1(void) {
    uint32_t x = 14u;
    uint32_t a = add_unsigned_bits(x, 1u, 4);
    uint32_t b = add_unsigned_bits(a, 1u, 4);
    uint32_t c = add_unsigned_bits(b, 1u, 4);
    printf("Ex1 (4-bit sequence from 14): %u %u %u\n", a, b, c); /* 15 0 1 */
}

/* Exercise 2: Does 0.1 added ten times equal exactly 1.0? */
static void exercise_2(void) {
    double s = repeat_add(0.1, 10);
    printf("Ex2 (ten*0.1): sum=%.17g, eq1=%s, almost=%s\n",
           s, (s == 1.0) ? "true" : "false",
           almost_equal(s, 1.0, 1e-9) ? "true" : "false");
}

/* Exercise 3: 8-bit signed addition demos: 127+1 and -1+(-1) */
static void exercise_3(void) {
    int32_t a = add_signed_bits(127, 1, 8);    /* expect -128 */
    int32_t b = add_signed_bits(-1, -1, 8);    /* expect -2   */
    printf("Ex3 (signed adds): %d %d\n", a, b);
}

/* Exercise 4: Unsigned 16-bit wrap for 65535 + 1 -> 0 */
static void exercise_4(void) {
    uint32_t r = add_unsigned_bits(65535u, 1u, 16);
    printf("Ex4 (u16 65535+1): %u\n", r);
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    return 0;
}
