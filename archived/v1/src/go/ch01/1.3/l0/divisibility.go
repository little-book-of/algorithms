package main

func IsDivisible(a, b int) bool {
	if b == 0 {
		return false // undefined; keep it simple for L0
	}
	return a%b == 0
}

func LastDecimalDigit(n int) int {
	if n < 0 {
		n = -n
	}
	return n % 10
}

func DivisibleBy10(n int) bool {
	return LastDecimalDigit(n) == 0
}
