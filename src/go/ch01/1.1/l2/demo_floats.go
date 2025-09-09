package main

import (
	"fmt"
	"math"
)

func main() {
	// Rounding surprise
	a := 0.1 + 0.2
	fmt.Printf("0.1 + 0.2 = %.17g\n", a)
	fmt.Println("Equal to 0.3?", a == 0.3)

	// Special values
	posInf := math.Inf(+1)
	nan := math.NaN()
	fmt.Println("Infinity:", posInf)
	fmt.Println("NaN == NaN?", nan == nan) // always false

	// Epsilon & ULP near 1.0
	fmt.Println("ULP(1.0):", ULP(1.0))
	nextUp := math.Nextafter(1.0, math.Inf(+1))
	nextDown := math.Nextafter(1.0, math.Inf(-1))
	fmt.Println("Nextafter(1.0, +inf):", nextUp)
	fmt.Println("Nextafter(1.0, -inf):", nextDown)
	fmt.Println("ULP around 1.0 (nextUp - 1.0):", nextUp-1.0)

	// Subnormal example
	tiny := math.SmallestNonzeroFloat64 // first positive subnormal
	fmt.Printf("First subnormal > 0: %.17g\n", tiny)
	fmt.Println("Classify tiny:", Classify(tiny))

	// Decompose a few values
	fmt.Println(ComponentsPretty(1.0))
	fmt.Println(ComponentsPretty(0.0))
	fmt.Println(ComponentsPretty(math.Inf(+1)))
	fmt.Println(ComponentsPretty(math.NaN()))
}
