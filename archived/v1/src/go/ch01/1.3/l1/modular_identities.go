package main

// normalize x into [0, m-1] for m > 0
func modNorm(x, m int64) int64 {
	r := x % m
	if r < 0 {
		r += m
	}
	return r
}

func ModAdd(a, b, m int64) int64 {
	a = modNorm(a, m)
	b = modNorm(b, m)
	return modNorm(a+b, m)
}

func ModSub(a, b, m int64) int64 {
	a = modNorm(a, m)
	b = modNorm(b, m)
	return modNorm(a-b, m)
}

func ModMul(a, b, m int64) int64 {
	a = modNorm(a, m)
	b = modNorm(b, m)
	// Use 128-bit math via big.Int if you expect overflow; for teaching:
	// do the reduction early when possible.
	return modNorm(a*b, m)
}