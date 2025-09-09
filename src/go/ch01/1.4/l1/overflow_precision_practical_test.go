package main

import (
	"math"
	"testing"
)

func TestInt32Ops(t *testing.T) {
	if _, ok := AddI32Checked(Int32Max, 1); ok {
		t.Fatal("expected overflow for AddI32Checked")
	}
	if got := AddI32Wrapping(Int32Max, 1); got != Int32Min {
		t.Fatalf("wrapping i32: got %d want %d", got, Int32Min)
	}
	if got := AddI32Saturating(Int32Max, 1); got != Int32Max {
		t.Fatalf("saturating i32: got %d want %d", got, Int32Max)
	}
	if _, ok := MulI32Checked(100000, 100000); ok {
		t.Fatal("mul i32 should overflow")
	}
	if got, ok := MulI32Checked(46341, 46341); !ok || got != 2147488281 {
		t.Fatalf("mul i32 widen: got %d ok=%v", got, ok)
	}
}

func TestInt64Ops(t *testing.T) {
	if _, ok := AddI64Checked(Int64Max, 1); ok {
		t.Fatal("expected overflow for AddI64Checked")
	}
	if AddI64Saturating(Int64Max, 1) != Int64Max {
		t.Fatal("saturating i64 max")
	}
	if AddI64Wrapping(Int64Max, 1) != Int64Min {
		t.Fatal("wrapping i64 should wrap to min")
	}
	if _, ok := MulI64Checked(Int64Max, 2); ok {
		t.Fatal("mul i64 should overflow")
	}
}

func TestFloatUtils(t *testing.T) {
	x := 0.1 + 0.2
	if x == 0.3 {
		t.Fatal("expected 0.1+0.2 != 0.3 in binary")
	}
	if !AlmostEqual(x, 0.3, 1e-12, 1e-12) {
		t.Fatal("AlmostEqual should hold")
	}
	// Summation stability (values close)
	n := 50000
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = 1.0 / float64(i+1)
	}
	p, k := PairwiseSum(xs), KahanSum(xs)
	if math.Abs(p-k) > 1e-10 {
		t.Fatalf("pairwise and kahan differ too much: %g", p-k)
	}
	eps := MachineEpsilon()
	if 1.0+eps == 1.0 {
		t.Fatal("machine epsilon probe failed")
	}
	if ULPDiff(1.0, math.Nextafter(1.0, 2.0)) != 1 {
		t.Fatal("ULP step should be 1")
	}
}

func TestMoney(t *testing.T) {
	c, err := ParseDollarsToCents("12.34")
	if err != nil || c != 1234 {
		t.Fatalf("parse cents 12.34 -> %d (%v)", c, err)
	}
	c, _ = ParseDollarsToCents("-0.07")
	if c != -7 {
		t.Fatalf("parse -0.07 -> %d", c)
	}
	c, _ = ParseDollarsToCents("1.239") // truncate
	if c != 123 {
		t.Fatalf("parse 1.239 -> %d", c)
	}
	a := &Ledger{BalanceCents: 1000}
	b := &Ledger{}
	_ = a.TransferTo(b, 335)
	if FormatCents(a.BalanceCents) != "6.65" || FormatCents(b.BalanceCents) != "3.35" {
		t.Fatal("ledger transfer formatting wrong")
	}
}

func TestExercises(t *testing.T) {
	wrap, chk, sat := Ex1Policy()
	if wrap != Int32Min || chk != "Overflow" || sat != Int32Max {
		t.Fatalf("Ex1Policy got wrap=%d chk=%s sat=%d", wrap, chk, sat)
	}
	sum, eq1, almost := Ex2CompareFloats()
	if eq1 || !almost {
		t.Fatal("Ex2CompareFloats expected eq=false almost=true")
	}
	p, k := Ex3Summation()
	if math.Abs(p-k) > 1e-10 {
		t.Fatal("Ex3Summation pairwise vs kahan mismatch")
	}
	a, b := Ex4Ledger()
	if a != "12.00" || b != "0.34" {
		t.Fatalf("Ex4Ledger got %s %s", a, b)
	}
}
