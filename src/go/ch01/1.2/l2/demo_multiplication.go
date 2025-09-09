package main

import "fmt"

// naiveMul just uses Go's built-in operator as a baseline.
func naiveMul(x, y int) int { return x * y }

func main() {
	a, b := 1234, 5678
	fmt.Println("Naive multiplication:", naiveMul(a, b))
	fmt.Println("Builtin multiplication:", a*b)

	// Run other demos
	runKaratsubaDemo()
	runModexpDemo()
	runExercises()
}
