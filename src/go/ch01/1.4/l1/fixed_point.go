package main

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// ParseDollarsToCents parses "123", "123.4", "123.45", "-0.07" into cents (int64).
// Extra fractional digits are truncated by policy.
func ParseDollarsToCents(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, errors.New("empty money literal")
	}
	sign := int64(1)
	if s[0] == '+' || s[0] == '-' {
		if s[0] == '-' {
			sign = -1
		}
		s = s[1:]
		if s == "" {
			return 0, errors.New("bad money literal")
		}
	}

	whole := int64(0)
	i := 0
	for i < len(s) && unicode.IsDigit(rune(s[i])) {
		d := int64(s[i] - '0')
		// whole = whole*10 + d (checked with 64-bit bounds)
		if z, ok := MulI64Checked(whole, 10); ok {
			if z2, ok2 := AddI64Checked(z, d); ok2 {
				whole = z2
			} else {
				return 0, errors.New("overflow parsing whole")
			}
		} else {
			return 0, errors.New("overflow parsing whole")
		}
		i++
	}

	d1, d2 := int64(0), int64(0)
	if i < len(s) && s[i] == '.' {
		i++
		if i < len(s) && unicode.IsDigit(rune(s[i])) {
			d1 = int64(s[i] - '0')
			i++
		}
		if i < len(s) && unicode.IsDigit(rune(s[i])) {
			d2 = int64(s[i] - '0')
			i++
		}
		// ignore extra fractional digits
		for i < len(s) && unicode.IsDigit(rune(s[i])) {
			i++
		}
	}

	if strings.TrimSpace(s[i:]) != "" {
		return 0, errors.New("trailing junk in money literal")
	}

	cents := int64(0)
	if z, ok := MulI64Checked(whole, 100); ok {
		if z2, ok2 := AddI64Checked(z, d1*10+d2); ok2 {
			cents = z2
		} else {
			return 0, errors.New("overflow combining cents")
		}
	} else {
		return 0, errors.New("overflow combining cents")
	}

	return sign * cents, nil
}

// FormatCents prints cents to "[-]W.DD".
func FormatCents(c int64) string {
	sign := ""
	if c < 0 {
		sign = "-"
		c = -c
	}
	return fmt.Sprintf("%s%d.%02d", sign, c/100, c%100)
}

type Ledger struct {
	BalanceCents int64
}

func (l *Ledger) Deposit(cents int64) error {
	if z, ok := AddI64Checked(l.BalanceCents, cents); ok {
		l.BalanceCents = z
		return nil
	}
	return errors.New("overflow on deposit")
}

func (l *Ledger) Withdraw(cents int64) error {
	if cents > l.BalanceCents {
		return errors.New("insufficient funds")
	}
	l.BalanceCents -= cents
	return nil
}

func (l *Ledger) TransferTo(other *Ledger, cents int64) error {
	if err := l.Withdraw(cents); err != nil {
		return err
	}
	if err := other.Deposit(cents); err != nil {
		// rollback on deposit failure
		l.BalanceCents += cents
		return err
	}
	return nil
}
