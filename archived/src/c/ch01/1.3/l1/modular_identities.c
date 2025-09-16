#include "modular_identities.h"
#include <stdlib.h> /* llabs */

static int64_t pos_mod(int64_t x, int64_t m) {
    int64_t r = x % m;
    return (r < 0) ? (r + m) : r;
}

int64_t mod_norm(int64_t x, int64_t m) {
    return pos_mod(x, m);
}

int64_t mod_add64(int64_t a, int64_t b, int64_t m) {
    a = pos_mod(a, m);
    b = pos_mod(b, m);
    int64_t s = a + b;
    return pos_mod(s, m);
}

int64_t mod_sub64(int64_t a, int64_t b, int64_t m) {
    a = pos_mod(a, m);
    b = pos_mod(b, m);
    int64_t s = a - b;
    return pos_mod(s, m);
}

int64_t mod_mul64(int64_t a, int64_t b, int64_t m) {
    a = pos_mod(a, m);
    b = pos_mod(b, m);
    __int128 t = (__int128)a * (__int128)b; /* reduce overflow risk */
    return (int64_t)pos_mod((int64_t)(t % m), m);
}