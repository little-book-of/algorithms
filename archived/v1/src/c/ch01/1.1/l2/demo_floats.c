#include <stdio.h>
#include <math.h>
#include <float.h>
#include "ieee754_utils.h"

int main(void) {
    /* 0.1 + 0.2 surprise */
    double a = 0.1 + 0.2;
    printf("0.1 + 0.2 = %.17g\n", a);
    printf("Equal to 0.3? %s\n", (a == 0.3) ? "true" : "false");

    /* Special values */
    double pos_inf = INFINITY;
    double nan = NAN;
    printf("Infinity: %g\n", pos_inf);
    printf("NaN == NaN? %s\n", (nan == nan) ? "true" : "false");

    /* Epsilon & ULP near 1.0 */
    printf("DBL_EPSILON: %.17g\n", DBL_EPSILON);
    printf("ULP(1.0): %.17g\n", ulp(1.0));

    /* nextafter around 1.0 */
    double next_up = nextafter(1.0, INFINITY);
    double next_down = nextafter(1.0, -INFINITY);
    printf("nextafter(1.0, +inf): %.17g\n", next_up);
    printf("nextafter(1.0, -inf): %.17g\n", next_down);
    printf("ULP around 1.0 (next_up - 1.0): %.17g\n", next_up - 1.0);

    /* Subnormal example */
    double tiny = nextafter(0.0, 1.0); /* first positive subnormal */
    printf("First subnormal > 0: %.17g\n", tiny);
    printf("Classify tiny: %s\n", (classify_double(tiny) == FP_SUBNORMAL) ? "subnormal" : "other");

    /* Decompose some values */
    char buf[160];
    fields_pretty(1.0, buf, sizeof buf);
    printf("%s\n", buf);
    fields_pretty(0.0, buf, sizeof buf);
    printf("%s\n", buf);
    fields_pretty(INFINITY, buf, sizeof buf);
    printf("%s\n", buf);
    fields_pretty(NAN, buf, sizeof buf);
    printf("%s\n", buf);

    return 0;
}
