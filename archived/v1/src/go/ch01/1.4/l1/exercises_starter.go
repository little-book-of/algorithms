package main

import "fmt"

// Ex1Policy: results of (Int32Max + 1) under wrapping / checked / saturating.
func Ex1Policy() (wrap int32, checked string, sat int32) {
	wrap = AddI32Wrapping(Int32Max, 1)
	if _, ok := AddI32Checked(Int32Max, 1); !ok {
		checked = "Overflow"
	} else {
		checked = "OK"
	}
	sat = AddI32Saturating(Int32Max, 1)
	return
}

func Ex2CompareFloats() (sum float64, eq1, almost bool) {
	sum = 0.1 + 0.2
	return sum, sum == 0.3, AlmostEqual(sum, 0.3, 1e-12, 1e-12)
}

func Ex3Summation() (pairwise, kahan float64) {
	n := 100000
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = 1.0 / float64(i+1)
	}
	return PairwiseSum(xs), KahanSum(xs)
}

func Ex4Ledger() (aStr, bStr string) {
	aC, _ := ParseDollarsToCents("12.34")
	bC, _ := ParseDollarsToCents("0.00")
	amt, _ := ParseDollarsToCents("0.34")
	a := &Ledger{BalanceCents: aC}
	b := &Ledger{BalanceCents: bC}
	_ = a.TransferTo(b, amt)
	return FormatCents(a.BalanceCents), FormatCents(b.BalanceCents)
}

func runExercises() {
	wrap, chk, sat := Ex1Policy()
	fmt.Println("Exercise 1 (policy): wrap=", wrap, "checked=", chk, "sat=", sat)
	sum, eq1, almost := Ex2CompareFloats()
	fmt.Printf("Exercise 2 (float compare): sum=%.17g eq=%v almost=%v\n", sum, eq1, almost)
	p, k := Ex3Summation()
	fmt.Printf("Exercise 3 (pairwise vs kahan): %.15f %.15f\n", p, k)
	a, b := Ex4Ledger()
	fmt.Println("Exercise 4 (ledger):", a, b)
}
