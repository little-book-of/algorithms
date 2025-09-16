package main

// DivIdentity returns (q, r) such that n = d*q + r, with r having Go's sign convention.
func DivIdentity(n, d int) (int, int) {
	// caller must ensure d != 0 for meaningful math; Go will panic on division by zero.
	return n / d, n % d
}

// WeekShift wraps 0..6 (0=Mon) and returns the day index after shift days.
// Handles negative inputs robustly.
func WeekShift(start, shift int) int {
	m := 7
	t := ((start % m) + m) % m
	s := ((shift % m) + m) % m
	return (t + s) % m
}

// PowMod computes (base^exp) % mod via binary exponentiation.
func PowMod(base, exp, mod int) int {
	if mod == 1 {
		return 0
	}
	res := 1 % mod
	b := ((base % mod) + mod) % mod
	e := exp
	for e > 0 {
		if e&1 == 1 {
			res = (res * b) % mod
		}
		b = (b * b) % mod
		e >>= 1
	}
	return res
}
