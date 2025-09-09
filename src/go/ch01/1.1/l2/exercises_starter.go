package main

import (
	"fmt"
	"math"
)

func ex1BitsOfOne() (sign uint64, expRaw uint64, mant uint64, unbiased int64) {
	f := Decompose(1.0)
	return f.Sign, f.ExponentRaw, f.Mantissa, UnbiasedExponent(f.ExponentRaw)
}

func ex2ULPNearOne() (ulp float64, nextUp float64) {
	next := math.Nextafter(1.0, math.Inf(+1))
	return next - 1.0, next
}

func ex3Classify() [][2]string {
	vals := []float64{0.0, 1.0, math.Inf(+1), math.NaN(), math.SmallestNonzeroFloat64}
	out := make([][2]string, 0, len(vals))
	for _, v := range vals {
		out = append(out, [2]string{fmt.Sprintf("%.17g", v), kindToString(Classify(v))})
	}
	return out
}

func ex4Roundtrip(x float64) (bits uint64, y float64) {
	b := Float64ToBits(x)
	return b, BitsToFloat64(b)
}

func ex5BuildInfAndNaN() (inf float64, nanEqual bool) {
	finf := FPFields{Sign: 0, ExponentRaw: (1<<ExpBits - 1), Mantissa: 0}
	fnan := FPFields{Sign: 0, ExponentRaw: (1<<ExpBits - 1), Mantissa: 1}
	inf = Build(finf)
	nan := Build(fnan)
	return inf, (nan == nan)
}

func runExercises() {
	// show exact decimal demo too
	decimalExactDemo()

	s, e, m, ub := ex1BitsOfOne()
	fmt.Printf("Ex1: sign=%d exp_raw=%d mantissa=0x%013x (unbiased=%d)\n", s, e, m, ub)

	ul, next := ex2ULPNearOne()
	fmt.Printf("Ex2: ULP(1.0)=%.17g nextUp=%.17g\n", ul, next)

	for _, pair := range ex3Classify() {
		fmt.Printf("Ex3: %s -> %s\n", pair[0], pair[1])
	}

	bits, y := ex4Roundtrip(3.5)
	fmt.Printf("Ex4: 3.5 bits=0x%016x back=%.17g\n", bits, y)

	inf, nanEq := ex5BuildInfAndNaN()
	fmt.Printf("Ex5: built +inf=%g, built nan==nan? %v\n", inf, nanEq)
}

func kindToString(k FPKind) string {
	switch k {
	case FPNan:
		return "nan"
	case FPInf:
		return "inf"
	case FPZero:
		return "zero"
	case FPSubnormal:
		return "subnormal"
	default:
		return "normal"
	}
}
