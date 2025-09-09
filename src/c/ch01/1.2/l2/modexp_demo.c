#include <stdio.h>
#include "modexp.h"

int main(void) {
    printf("modexp(7,128,13) = %lld\n", modexp(7,128,13));
    printf("Expected = 3\n");
    return 0;
}
