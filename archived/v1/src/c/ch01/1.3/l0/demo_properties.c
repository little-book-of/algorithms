#include <stdio.h>
#include <stdint.h>
#include "parity.h"
#include "divisibility.h"
#include "remainder.h"

int main(void) {
    puts("Parity:");
    for (int64_t n : (int64_t[]){10, 7}) {
        printf("  %lld is_even? %s  is_odd? %s  (bit) %s\n",
               (long long)n,
               is_even(n) ? "true" : "false",
               is_odd(n)  ? "true" : "false",
               parity_bit(n));
    }

    puts("\nDivisibility:");
    printf("  12 divisible by 3? %s\n", is_divisible(12,3) ? "true" : "false");
    printf("  14 divisible by 5? %s\n", is_divisible(14,5) ? "true" : "false");

    puts("\nRemainders & identity:");
    int64_t q, r;
    div_identity(17, 5, &q, &r);
    printf("  17 = 5*%lld + %lld\n", (long long)q, (long long)r);

    puts("\nClock (7-day week):");
    /* 0=Mon, 5=Sat; Sat + 10 -> Mon(1) */
    printf("  Saturday(5) + 10 days -> %d\n", week_shift(5, 10));

    puts("\nLast digit of 2^15:");
    printf("  %lld\n", (long long)powmod(2, 15, 10));
    return 0;
}
