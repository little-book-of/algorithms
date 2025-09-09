#include <stdio.h>

int exercise_1(void) { return 326 + 589; }
int exercise_2(void) { return 704 - 259; }
int exercise_3(void) { return 38 * 12; }
void exercise_4(int *q, int *r) { *q = 123 / 7; *r = 123 % 7; }

int main(void) {
    printf("Exercise 1 (326+589): %d\n", exercise_1());
    printf("Exercise 2 (704-259): %d\n", exercise_2());
    printf("Exercise 3 (38*12): %d\n", exercise_3());
    int q, r;
    exercise_4(&q, &r);
    printf("Exercise 4 (123//7, 123%%7): %d %d\n", q, r);
    return 0;
}
