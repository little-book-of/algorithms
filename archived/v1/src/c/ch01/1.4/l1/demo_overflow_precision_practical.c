#include <stdio.h>
#include <inttypes.h>
#include <math.h>
#include "int_arith.h"
#include "float_num.h"
#include "fixed_point.h"

int main(void) {
    puts("=== Integers: checked / wrapping / saturating ===");
    printf("int32 max: %d\n", INT32_MAX_C);

    int32_t out32;
    if (!add_i32_checked(INT32_MAX_C, 1, &out32))
        puts("checked add (i32): overflow trapped ✔");

    printf("wrapping add (i32): %d\n", add_i32_wrapping(INT32_MAX_C, 1));     /* INT32_MIN */
    printf("saturating add (i32): %d\n", add_i32_saturating(INT32_MAX_C, 1)); /* INT32_MAX */

    printf("wrapping add (i64): %" PRId64 "\n", add_i64_wrapping(9223372036854775607LL, 100));
    printf("saturating add (i64): %" PRId64 "\n", add_i64_saturating(9223372036854775700LL, 200));

    puts("\n=== Floats: compare, sum, epsilon ===");
    double x = 0.1 + 0.2;
    printf("0.1 + 0.2 = %.17g\n", x);
    printf("Direct equality with 0.3? %s\n", (x == 0.3) ? "true" : "false");
    printf("almost_equal(x, 0.3): %s\n", almost_equal(x, 0.3, 1e-12, 1e-12) ? "true" : "false");

    /* harmonic(1..1e5) for demo; pairwise & Kahan (skip naive) */
    size_t n = 100000;
    double *arr = (double *)malloc(n * sizeof(double));
    for (size_t i = 0; i < n; ++i) arr[i] = 1.0 / (double)(i + 1);
    printf("pairwise_sum: %.15f\n", pairwise_sum(arr, n));
    printf("kahan_sum   : %.15f\n", kahan_sum(arr, n));
    free(arr);

    printf("machine epsilon: %.20g\n", machine_epsilon());
    printf("ULP(1.0, nextafter(1.0, 2.0)) = %" PRIu64 "\n", ulp_diff(1.0, nextafter(1.0, 2.0)));

    puts("\n=== Fixed-point money (int cents) ===");
    int64_t ca, cdep, c335;
    parse_dollars_to_cents("10.00", &ca);
    parse_dollars_to_cents("0.05", &cdep);
    parse_dollars_to_cents("3.35", &c335);

    Ledger A, B; ledger_init(&A, ca); ledger_init(&B, 0);
    ledger_transfer(&A, &B, c335);
    char bufA[32], bufB[32];
    format_cents(A.balance_cents, bufA, sizeof bufA);
    format_cents(B.balance_cents, bufB, sizeof bufB);
    printf("A: %s B: %s\n", bufA, bufB);
    ledger_deposit(&A, cdep);
    format_cents(A.balance_cents, bufA, sizeof bufA);
    printf("A after 5 cents deposit: %s\n", bufA);

    return 0;
}
