#include <stdio.h>

int main(void) {
    int a = 478, b = 259;
    printf("Addition: %d + %d = %d\n", a, b, a + b);

    a = 503; b = 78;
    printf("Subtraction: %d - %d = %d\n", a, b, a - b);

    a = 214; b = 3;
    printf("Multiplication: %d * %d = %d\n", a, b, a * b);

    int n = 47, d = 5;
    int q = n / d;
    int r = n % d;
    printf("Division: %d / %d -> quotient %d, remainder %d\n", n, d, q, r);

    return 0;
}
