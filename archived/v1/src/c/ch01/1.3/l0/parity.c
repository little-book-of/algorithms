#include "parity.h"

bool is_even(int64_t n) { return (n % 2) == 0; }
bool is_odd (int64_t n) { return (n % 2) != 0; }

const char *parity_bit(int64_t n) {
    return (n & 1) ? "odd" : "even";
}
