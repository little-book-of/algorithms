package main

import (
	"fmt"
	"math"
)

const (
	MantBits = 52
	ExpBits  = 11
	ExpBias  = 1023
)

type FPKind int

const (
	FPNan FPKind = iota
	FPInf
	FPZero
	FPSubnormal
	FPNormal
)

type FPFields struct {
	Sign        uint64 // 0 or 1
	ExponentRaw uint64 // 11 bits
	Mantissa    uint64 // 52 bits
}

func Float64ToBits(x float64) uint64 { return math.Float64bits(x) }
func BitsToFloat64(b uint64) float64 { return math.Float64frombits(b) }

func Decompose(x float64) FPFields {
	b := math.Float64bits(x)
	sign := (b >> 63) & 0x1
	exp := (b >> MantBits) & ((1 << ExpBits) - 1)
	man := b & ((1 << MantBits) - 1)
	return FPFields{Sign: sign, ExponentRaw: exp, Mantissa: man}
}

func Build(f FPFields) float64 {
	sign := (f.Sign & 1) << 63
	exp := (f.ExponentRaw & ((1 << ExpBits) - 1)) << MantBits
	man := f.Mantissa & ((1 << MantBits) - 1)
	return math.Float64frombits(sign | exp | man)
}

func Classify(x float64) FPKind {
	f := Decompose(x)
	switch {
	case f.ExponentRaw == (1<<ExpBits - 1):
		if f.Mantissa != 0 {
			return FPNan
		}
		return FPInf
	case f.ExponentRaw == 0:
		if f.Mantissa == 0 {
			return FPZero
		}
		return FPSubnormal
	default:
		return FPNormal
	}
}

func ComponentsPretty(x float64) string {
	f := Decompose(x)
	var kind string
	switch Classify(x) {
	case FPNan:
		kind = "nan"
	case FPInf:
		kind = "inf"
	case FPZero:
		kind = "zero"
	case FPSubnormal:
		kind = "subnormal"
	default:
		kind = "normal"
	}
	return fmt.Sprintf("x=%g kind=%s sign=%d exponent_raw=%d mantissa=0x%013x",
		x, kind, f.Sign, f.ExponentRaw, f.Mantissa)
}

func UnbiasedExponent(expRaw uint64) int64 {
	return int64(expRaw) - ExpBias
}

// ULP returns the spacing to the next representable float in +∞ direction.
func ULP(x float64) float64 {
	return math.Abs(math.Nextafter(x, math.Inf(+1)) - x)
}
