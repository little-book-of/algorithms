#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "parity.h"
#include "divisibility.h"
#include "remainder.h"

const char *exercise_1(int64_t n) { return is_even(n) ? "even" : "odd"; }
bool        exercise_2(void)      { return is_divisible(91, 7); }
int64_t     exercise_3(void)      { int64_t q,r; div_identity(100,9,&q,&r); return r; }
int         exercise_4(void)      { return week_shift(5, 10); }     /* Sat + 10 -> Mon(1) */
int64_t     exercise_5(void)      { return powmod(2, 15, 10); }     /* last digit of 2^15 */

int main(void) {
    printf("Exercise 1 (42 parity): %s\n", exercise_1(42));
    printf("Exercise 2 (91 divisible by 7?): %s\n", exercise_2() ? "true" : "false");
    printf("Exercise 3 (100 %% 9): %lld\n", (long long)exercise_3());
    printf("Exercise 4 (Sat+10): %d\n", exercise_4());
    printf("Exercise 5 (last digit of 2^15): %lld\n", (long long)exercise_5());
    return 0;
}
