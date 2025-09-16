#include <stdio.h>
#include "karatsuba.h"

int main(void) {
    long long a = 1234, b = 5678;
    long long res = karatsuba(a, b);
    printf("Karatsuba result: %lld\n", res);
    printf("Expected: %lld\n", a * b);
    return 0;
}
