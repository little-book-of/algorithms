package main

func abs64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

// GCD via Euclid's algorithm (non-negative result).
func GCD(a, b int64) int64 {
	a, b = abs64(a), abs64(b)
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// LCM via gcd; lcm(0, b) = 0 by convention.
func LCM(a, b int64) int64 {
	if a == 0 || b == 0 {
		return 0
	}
	g := GCD(a, b)
	// divide before multiply to reduce overflow risk
	return abs64((a / g) * b)
}