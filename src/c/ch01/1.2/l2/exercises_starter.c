#include <stdio.h>
#include "karatsuba.h"
#include "modexp.h"

void exercise_1(void) {
    long long res = karatsuba(31415926LL, 27182818LL);
    printf("Exercise 1 (Karatsuba 31415926*27182818): %lld\n", res);
}

void exercise_2(void) {
    long long res = modexp(5,117,19);
    printf("Exercise 2 (modexp 5^117 mod 19): %lld\n", res);
}

void exercise_3(void) {
    long long a = 12345, b = 67890;
    long long res = karatsuba(a,b);
    printf("Exercise 3 (check Karatsuba correctness): %s\n", 
           (res == a*b) ? "true" : "false");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    return 0;
}
