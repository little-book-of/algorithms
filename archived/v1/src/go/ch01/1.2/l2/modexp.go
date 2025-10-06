package main

// ModExp computes (a^b) mod m using square-and-multiply.
func ModExp(a, b, m int) int {
	if m == 1 {
		return 0
	}
	res := 1
	base := a % m
	exp := b
	for exp > 0 {
		if exp&1 == 1 {
			res = (res * base) % m
		}
		base = (base * base) % m
		exp >>= 1
	}
	return res
}
