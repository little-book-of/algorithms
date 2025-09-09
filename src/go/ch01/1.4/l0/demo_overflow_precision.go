package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Integer overflow (8-bit) ===")
	fmt.Println("255 + 1 (unsigned, wrap) ->", AddUnsignedBits(255, 1, 8)) // 0
	fmt.Println("127 + 1 (signed, wrap)   ->", AddSignedBits(127, 1, 8))   // -128

	r, cf, of := AddWithFlags8(200, 100)
	fmt.Printf("200 + 100 -> result=%d (unsigned), CF=%v, OF=%v\n", r, cf, of)
	fmt.Println("Interpret result as signed:", ToSignedBits(uint32(r), 8))

	fmt.Println("\n=== Floating-point surprises ===")
	x := 0.1 + 0.2
	fmt.Printf("0.1 + 0.2 = %.17g\n", x) // 0.30000000000000004
	fmt.Println("Direct equality with 0.3?", x == 0.3)
	fmt.Println("Using epsilon:", AlmostEqual(x, 0.3, 1e-9))

	fmt.Println("\nRepeat add 0.1 ten times:")
	s := RepeatAdd(0.1, 10)
	fmt.Printf("sum = %.17g, equal to 1.0? %v, almost_equal? %v\n",
		s, s == 1.0, AlmostEqual(s, 1.0, 1e-9))

	fmt.Println("\nMix large and small:")
	a := MixLargeSmall(1e16, 1.0)
	fmt.Printf("1e16 + 1.0 = %.17g (often unchanged)\n", a)

	fmt.Println("\nNaive sum may drift slightly:")
	nums := make([]float64, 10)
	for i := range nums {
		nums[i] = 0.1
	}
	fmt.Printf("SumNaive([0.1]*10) = %.17g\n", SumNaive(nums))

	// Optionally run the small exercise demo:
	runExercises()
}
