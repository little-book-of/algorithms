package main

import (
	"math"
	"testing"
)

func TestDecomposeOne(t *testing.T) {
	f := Decompose(1.0)
	if f.Sign != 0 {
		t.Fatalf("sign(1.0) want 0 got %d", f.Sign)
	}
	if f.ExponentRaw != 1023 {
		t.Fatalf("exp_raw(1.0) want 1023 got %d", f.ExponentRaw)
	}
	if f.Mantissa != 0 {
		t.Fatalf("mantissa(1.0) want 0 got %x", f.Mantissa)
	}
	if UnbiasedExponent(f.ExponentRaw) != 0 {
		t.Fatalf("unbiased exponent want 0")
	}
}

func TestRoundtrip(t *testing.T) {
	x := 3.5
	b := Float64ToBits(x)
	y := BitsToFloat64(b)
	if y != x {
		t.Fatalf("roundtrip %v -> %x -> %v", x, b, y)
	}
}

func TestClassify(t *testing.T) {
	if Classify(0.0) != FPZero {
		t.Fatal("0.0 should be zero")
	}
	if Classify(1.0) != FPNormal {
		t.Fatal("1.0 should be normal")
	}
	if Classify(math.Inf(+1)) != FPInf {
		t.Fatal("+Inf should be inf")
	}
	if Classify(math.SmallestNonzeroFloat64) != FPSubnormal {
		t.Fatal("Smallest nonzero should be subnormal")
	}
}

func TestULP(t *testing.T) {
	ul := ULP(1.0)
	nx := math.Nextafter(1.0, math.Inf(+1))
	if ul != nx-1.0 {
		t.Fatalf("ULP mismatch: got %g want %g", ul, nx-1.0)
	}
}
