package main

import "fmt"

// bin returns a binary string for a non-negative int using Go's formatting.
func bin(x int) string {
	return fmt.Sprintf("%b", x)
}

func demoBinary() {
	x, y := 0b1011, 0b0110 // 11 and 6
	fmt.Println("x =", bin(x), "(dec", x, ")")
	fmt.Println("y =", bin(y), "(dec", y, ")")

	sum := x + y
	fmt.Println("x + y =", bin(sum), "(dec", sum, ")")

	diff := x - y
	fmt.Println("x - y =", bin(diff), "(dec", diff, ")")
}
