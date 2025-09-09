#include "divisibility.h"
#include <stdlib.h>

bool is_divisible(int64_t a, int64_t b) {
    /* divisibility by zero is undefined */
    if (b == 0) return false;
    return (a % b) == 0;
}

int last_decimal_digit(int64_t n) {
    int64_t m = llabs(n);
    return (int)(m % 10);
}

bool divisible_by_10(int64_t n) {
    return last_decimal_digit(n) == 0;
}
