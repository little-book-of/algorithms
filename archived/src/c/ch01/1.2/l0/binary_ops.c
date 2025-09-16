#include <stdio.h>

/* Print an integer in binary (up to 32 bits). */
void print_binary(int x) {
    for (int i = 31; i >= 0; --i) {
        putchar((x & (1 << i)) ? '1' : '0');
    }
}

int main(void) {
    int x = 0b1011; /* 11 */
    int y = 0b0110; /* 6 */

    printf("x = "); print_binary(x); printf(" (%d)\n", x);
    printf("y = "); print_binary(y); printf(" (%d)\n", y);

    int sum = x + y;
    printf("x + y = "); print_binary(sum); printf(" (%d)\n", sum);

    int diff = x - y;
    printf("x - y = "); print_binary(diff); printf(" (%d)\n", diff);

    return 0;
}
