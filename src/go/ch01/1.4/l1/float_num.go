package main

import "math"

// AlmostEqual reports true if |x - y| <= max(rel*max(|x|,|y|), abs).
func AlmostEqual(x, y, rel, abs float64) bool {
	ax, ay := math.Abs(x), math.Abs(y)
	diff := math.Abs(x - y)
	tol := math.Max(rel*math.Max(ax, ay), abs)
	return diff <= tol
}

// KahanSum: compensated summation, great for small/medium slices.
func KahanSum(xs []float64) float64 {
	var s, c float64
	for _, x := range xs {
		y := x - c
		t := s + y
		c = (t - s) - y
		s = t
	}
	return s
}

// PairwiseSum: stable tree/pairwise summation (iterative).
func PairwiseSum(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		return 0
	}
	buf := make([]float64, n)
	copy(buf, xs)
	m := n
	for m > 1 {
		k := 0
		i := 0
		for i+1 < m {
			buf[k] = buf[i] + buf[i+1]
			k++
			i += 2
		}
		if i < m {
			buf[k] = buf[i]
			k++
		}
		m = k
	}
	return buf[0]
}

// MachineEpsilon returns the smallest eps with 1.0+eps != 1.0.
func MachineEpsilon() float64 {
	eps := 1.0
	for 1.0+eps/2.0 != 1.0 {
		eps /= 2.0
	}
	return eps
}

// ULPDiff returns the distance in representable steps between a and b.
func ULPDiff(a, b float64) uint64 {
	ua := math.Float64bits(a)
	ub := math.Float64bits(b)
	if int64(ua) < 0 {
		ua = 0x8000000000000000 - ua
	}
	if int64(ub) < 0 {
		ub = 0x8000000000000000 - ub
	}
	if ua > ub {
		return ua - ub
	}
	return ub - ua
}
