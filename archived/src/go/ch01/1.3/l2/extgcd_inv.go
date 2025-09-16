package main

import "errors"

// ExtendedGCD returns (g, x, y) such that a*x + b*y = g = gcd(a,b).
func ExtendedGCD(a, b int64) (g, x, y int64) {
	oldR, r := a, b
	oldS, s := int64(1), int64(0)
	oldT, t := int64(0), int64(1)
	for r != 0 {
		q := oldR / r
		oldR, r = r, oldR-q*r
		oldS, s = s, oldS-q*s
		oldT, t = t, oldT-q*t
	}
	return oldR, oldS, oldT
}

// InvMod returns x in [0,m-1] with (a*x) % m == 1; error if gcd(a,m) != 1.
func InvMod(a, m int64) (int64, error) {
	if m <= 0 {
		return 0, errors.New("modulus must be positive")
	}
	g, x, _ := ExtendedGCD(a, m)
	if g != 1 && g != -1 {
		return 0, errors.New("inverse does not exist; gcd(a,m) != 1")
	}
	// normalize
	x %= m
	if x < 0 {
		x += m
	}
	return x, nil
}