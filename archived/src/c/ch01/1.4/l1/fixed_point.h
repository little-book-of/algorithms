#ifndef FIXED_POINT_H
#define FIXED_POINT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Parse "123", "123.4", "123.45", "-0.07" into cents (int64). Truncate extra decimals. */
bool parse_dollars_to_cents(const char *s, int64_t *out_cents);

/* Format cents to "[-]W.DD" into buf. buflen >= 32 is plenty. */
void format_cents(int64_t cents, char *buf, size_t buflen);

typedef struct {
    int64_t balance_cents;
} Ledger;

void ledger_init(Ledger *L, int64_t cents);
bool ledger_deposit(Ledger *L, int64_t cents);  /* returns false on overflow */
bool ledger_withdraw(Ledger *L, int64_t cents); /* insufficient -> false */
bool ledger_transfer(Ledger *from, Ledger *to, int64_t cents);

#endif
