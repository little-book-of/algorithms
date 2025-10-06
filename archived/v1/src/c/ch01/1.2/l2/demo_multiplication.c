#include <stdio.h>

int naive_mul(int x, int y) {
    return x * y; // just builtin
}

int main(void) {
    int a = 1234, b = 5678;
    printf("Naive multiplication: %d\n", naive_mul(a, b));
    printf("Builtin multiplication: %d\n", a * b);
    return 0;
}
