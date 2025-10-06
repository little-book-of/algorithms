#include <stdio.h>
#include <stdbool.h>

void exercise_1(void) {
    int q = 100 / 9;
    int r = 100 % 9;
    printf("Exercise 1 (100//9,100%%9): %d %d\n", q, r);
}

void exercise_2(void) {
    int q = 123 / 11;
    int r = 123 % 11;
    printf("Exercise 2 (123//11,123%%11): %d %d\n", q, r);
}

void exercise_3(int n, int d) {
    int q = n / d, r = n % d;
    bool ok = (n == d * q + r);
    printf("Exercise 3 check (%d,%d): %s\n", n, d, ok ? "true" : "false");
}

void exercise_4(int n) {
    printf("Exercise 4 (n %% 16 via bitmask): %d -> %d\n", n, n & 15);
}

void exercise_5(int n) {
    printf("Exercise 5 (%d divisible by 7?): %s\n", n, (n % 7 == 0) ? "true" : "false");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3(200, 23);
    exercise_4(37);
    exercise_5(35);
    return 0;
}
