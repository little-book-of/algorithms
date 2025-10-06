#include "fixed_point.h"
#include "int_arith.h"
#include <ctype.h>
#include <stdio.h>
#include <string.h>

static const char* skip_ws(const char *s) {
    while (*s && isspace((unsigned char)*s)) ++s;
    return s;
}

bool parse_dollars_to_cents(const char *s, int64_t *out_cents) {
    if (!s || !out_cents) return false;
    s = skip_ws(s);
    int sign = 1;
    if (*s == '+' || *s == '-') { if (*s == '-') sign = -1; ++s; }

    /* whole part */
    if (!*s || (!isdigit((unsigned char)*s) && *s != '.')) return false;
    int64_t whole = 0;
    while (*s && isdigit((unsigned char)*s)) {
        int digit = *s - '0';
        int64_t tmp;
        if (!mul_i64_checked(whole, 10, &tmp)) return false;
        if (!add_i64_checked(tmp, digit, &whole)) return false;
        ++s;
    }

    /* fractional part */
    int d1 = 0, d2 = 0; int nd = 0;
    if (*s == '.') {
        ++s;
        if (*s && isdigit((unsigned char)*s)) { d1 = *s - '0'; ++s; ++nd; }
        if (*s && isdigit((unsigned char)*s)) { d2 = *s - '0'; ++s; ++nd; }
        /* ignore extra digits (truncate) */
        while (*s && isdigit((unsigned char)*s)) ++s;
    }

    s = skip_ws(s);
    if (*s != '\0') return false;

    int64_t cents = 0, tmp;
    if (!mul_i64_checked(whole, 100, &tmp)) return false;
    if (!add_i64_checked(tmp, (int64_t)(d1*10 + d2), &cents)) return false;
    if (sign < 0) cents = -cents;
    *out_cents = cents;
    return true;
}

void format_cents(int64_t cents, char *buf, size_t buflen) {
    int64_t abs = cents < 0 ? -cents : cents;
    int64_t dollars = abs / 100;
    int64_t frac = abs % 100;
    if (cents < 0)
        snprintf(buf, buflen, "-%lld.%02lld",
                 (long long)dollars, (long long)frac);
    else
        snprintf(buf, buflen, "%lld.%02lld",
                 (long long)dollars, (long long)frac);
}

void ledger_init(Ledger *L, int64_t cents) {
    if (L) L->balance_cents = cents;
}

bool ledger_deposit(Ledger *L, int64_t cents) {
    if (!L) return false;
    int64_t out;
    if (!add_i64_checked(L->balance_cents, cents, &out)) return false;
    L->balance_cents = out;
    return true;
}

bool ledger_withdraw(Ledger *L, int64_t cents) {
    if (!L) return false;
    if (cents > L->balance_cents) return false;
    L->balance_cents -= cents;
    return true;
}

bool ledger_transfer(Ledger *from, Ledger *to, int64_t cents) {
    if (!from || !to) return false;
    if (!ledger_withdraw(from, cents)) return false;
    if (!ledger_deposit(to, cents)) { /* rollback if deposit fails */
        from->balance_cents += cents;
        return false;
    }
    return true;
}
