package main

import (
	"fmt"
	"math"
)

func main() {
	fmt.Println("=== Integers: checked / wrapping / saturating ===")
	fmt.Println("int32 max:", Int32Max)

	if _, ok := AddI32Checked(Int32Max, 1); !ok {
		fmt.Println("checked add (i32): overflow trapped ✔")
	}
	fmt.Println("wrapping add (i32):", AddI32Wrapping(Int32Max, 1))     // -> Int32Min
	fmt.Println("saturating add (i32):", AddI32Saturating(Int32Max, 1)) // -> Int32Max

	fmt.Println("wrapping add (i64):", AddI64Wrapping(Int64Max-41, 100))
	fmt.Println("saturating add (i64):", AddI64Saturating(Int64Max-100, 200))

	fmt.Println("\n=== Floats: compare, sum, epsilon ===")
	x := 0.1 + 0.2
	fmt.Printf("0.1 + 0.2 = %.17g\n", x)
	fmt.Println("Direct equality with 0.3?", x == 0.3)
	fmt.Println("AlmostEqual(x, 0.3):", AlmostEqual(x, 0.3, 1e-12, 1e-12))

	// harmonic(1..1e5); compare pairwise vs Kahan
	n := 100000
	arr := make([]float64, n)
	for i := range arr {
		arr[i] = 1.0 / float64(i+1)
	}
	fmt.Printf("pairwise_sum: %.15f\n", PairwiseSum(arr))
	fmt.Printf("kahan_sum   : %.15f\n", KahanSum(arr))

	fmt.Println("machine epsilon:", MachineEpsilon())
	fmt.Println("ULP(1.0, nextafter(1.0, 2.0)) =", ULPDiff(1.0, math.Nextafter(1.0, 2.0)))

	fmt.Println("\n=== Fixed-point money (int cents) ===")
	ca, _ := ParseDollarsToCents("10.00")
	cdep, _ := ParseDollarsToCents("0.05")
	c335, _ := ParseDollarsToCents("3.35")
	a := &Ledger{BalanceCents: ca}
	b := &Ledger{BalanceCents: 0}
	_ = a.TransferTo(b, c335)
	fmt.Println("A:", FormatCents(a.BalanceCents), "B:", FormatCents(b.BalanceCents))
	_ = a.Deposit(cdep)
	fmt.Println("A after 5 cents deposit:", FormatCents(a.BalanceCents))
}
