package main

import "fmt"

func main() {
	// Addition
	a, b := 478, 259
	fmt.Printf("Addition: %d + %d = %d\n", a, b, a+b)

	// Subtraction
	a, b = 503, 78
	fmt.Printf("Subtraction: %d - %d = %d\n", a, b, a-b)

	// Multiplication
	a, b = 214, 3
	fmt.Printf("Multiplication: %d * %d = %d\n", a, b, a*b)

	// Integer division + remainder
	n, d := 47, 5
	q, r := n/d, n%d
	fmt.Printf("Division: %d / %d -> quotient %d, remainder %d\n", n, d, q, r)
}
