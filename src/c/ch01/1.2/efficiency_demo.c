#include <stdio.h>

int main(void) {
    int nums[] = {5, 12, 20};
    int size = sizeof(nums) / sizeof(nums[0]);
    for (int i = 0; i < size; i++) {
        int n = nums[i];
        printf("%d %% 8 = %d, bitmask = %d\n", n, n % 8, n & 7);
    }
    return 0;
}
