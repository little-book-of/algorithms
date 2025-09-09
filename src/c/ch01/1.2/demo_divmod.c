#include <stdio.h>

int main(void) {
    int n, d, q, r;

    n = 47; d = 5;
    q = n / d; r = n % d;
    printf("%d = %d*%d + %d\n", n, d, q, r);

    n = 23; d = 7;
    q = n / d; r = n % d;
    printf("%d = %d*%d + %d\n", n, d, q, r);

    n = 100; d = 9;
    q = n / d; r = n % d;
    printf("%d = %d*%d + %d\n", n, d, q, r);

    return 0;
}
