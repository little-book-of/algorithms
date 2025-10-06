#include <stdio.h>
#include <stdint.h>

/* Simple hash bucket mapping */
int hashing_example(int key, int size) {
    return key % size;
}

/* Cyclic wrap-around for weekdays (0–6) */
int cyclic_day(int start, int shift) {
    return (start + shift) % 7;
}

/* Modular exponentiation (square-and-multiply) */
int modexp(int a, int b, int m) {
    long long result = 1;
    long long base = a % m;
    int exp = b;

    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % m;
        }
        base = (base * base) % m;
        exp >>= 1;
    }
    return (int)result;
}

int main(void) {
    printf("Hashing: key=1234, size=10 -> %d\n", hashing_example(1234, 10));
    printf("Cyclic: Saturday(5)+4 -> %d\n", cyclic_day(5, 4));
    printf("modexp(7,128,13) = %d\n", modexp(7, 128, 13));
    return 0;
}
