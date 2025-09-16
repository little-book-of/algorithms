package main

import "fmt"

func main() {
	// Show the division identity n = d*q + r
	n, d := 47, 5
	q, r := n/d, n%d
	fmt.Printf("%d = %d*%d + %d\n", n, d, q, r)

	n, d = 23, 7
	q, r = n/d, n%d
	fmt.Printf("%d = %d*%d + %d\n", n, d, q, r)

	n, d = 100, 9
	q, r = n/d, n%d
	fmt.Printf("%d = %d*%d + %d\n", n, d, q, r)

	// Run the other demos
	runModuloExamples()
	runEfficiencyDemo()
	runExercises()
}
