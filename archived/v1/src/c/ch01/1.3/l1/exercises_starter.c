#include <stdio.h>
#include <stdint.h>
#include "gcd_lcm.h"
#include "modular_identities.h"
#include "fractions_utils.h"

int64_t exercise_1(void) { return gcd64(252, 198); }
int64_t exercise_2(void) { return lcm64(12, 18); }

void exercise_3(int64_t *lhs, int64_t *rhs) {
    *lhs = (37 + 85) % 12;
    *rhs = mod_add64(37, 85, 12);
}

frac64 exercise_4(void) { return reduce_fraction64(84, 126); }
int64_t exercise_5(void) { return lcm64(12, 18); }

int main(void) {
    printf("Exercise 1 gcd(252,198): %lld\n", (long long)exercise_1());
    printf("Exercise 2 lcm(12,18): %lld\n", (long long)exercise_2());
    int64_t lhs, rhs; exercise_3(&lhs, &rhs);
    printf("Exercise 3 modular add check: %lld %lld\n", (long long)lhs, (long long)rhs);
    frac64 f = exercise_4();
    printf("Exercise 4 reduce 84/126: %lld/%lld\n", (long long)f.num, (long long)f.den);
    printf("Exercise 5 smallest day multiple: %lld\n", (long long)exercise_5());
    return 0;
}