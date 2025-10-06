package main

import "errors"

// PhiU64 computes Euler’s totient φ(n) for n>=1 using trial factorization.
func PhiU64(n uint64) (uint64, error) {
	if n == 0 {
		return 0, errors.New("n must be positive")
	}
	result := n
	x := n
	var p uint64 = 2
	for p*p <= x {
		if x%p == 0 {
			for x%p == 0 {
				x /= p
			}
			result -= result / p
		}
		if p == 2 {
			p = 3
		} else {
			p += 2
		}
	}
	if x > 1 {
		result -= result / x
	}
	return result, nil
}