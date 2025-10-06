package main

// Karatsuba multiplies two non-negative integers using the Karatsuba method.
// Falls back to builtin for small values. Works for int range; for very large
// integers, prefer math/big (not covered here to keep the example focused).
func Karatsuba(x, y int) int {
	if x < 10 || y < 10 {
		return x * y
	}
	// choose split position by bit-length
	n := max(bitLen(x), bitLen(y))
	m := n / 2

	high1, low1 := x>>m, x-(x>>m)<<m
	high2, low2 := y>>m, y-(y>>m)<<m

	z0 := Karatsuba(low1, low2)
	z2 := Karatsuba(high1, high2)
	z1 := Karatsuba(low1+high1, low2+high2) - z0 - z2

	return (z2 << (2 * m)) + (z1 << m) + z0
}

func bitLen(v int) int {
	if v == 0 {
		return 0
	}
	l := 0
	for vv := v; vv > 0; vv >>= 1 {
		l++
	}
	return l
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
