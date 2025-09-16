package main

import "math"

// AlmostEqual reports true if |x - y| < eps (simple beginner-safe comparison).
func AlmostEqual(x, y, eps float64) bool {
	return math.Abs(x-y) < eps
}

// RepeatAdd adds 'value' to 0.0 'times' times (shows tiny drift for values like 0.1).
func RepeatAdd(value float64, times int) float64 {
	s := 0.0
	for i := 0; i < times; i++ {
		s += value
	}
	return s
}

// MixLargeSmall: adding a tiny number to a huge one may have no effect (limited precision).
func MixLargeSmall(large, small float64) float64 {
	return large + small
}

// SumNaive sums left-to-right; useful to demonstrate accumulated error on floats.
func SumNaive(xs []float64) float64 {
	s := 0.0
	for _, v := range xs {
		s += v
	}
	return s
}
