package main

type Fraction struct {
	Num int64
	Den int64
}

// ReduceFraction returns (num, den) in lowest terms with a canonical sign
// (denominator always positive). Panics if d == 0 to keep API simple here.
func ReduceFraction(n, d int64) Fraction {
	if d == 0 {
		panic("denominator cannot be zero")
	}
	g := GCD(n, d)
	num := n / g
	den := d / g
	if den < 0 {
		num, den = -num, -den
	}
	return Fraction{Num: num, Den: den}
}