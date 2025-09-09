package main

import "fmt"

func ex1Karatsuba() int {
	return Karatsuba(31415926, 27182818)
}

func ex2ModExp() int {
	return ModExp(5, 117, 19)
}

func ex3Check() bool {
	a, b := 12345, 67890
	return Karatsuba(a, b) == a*b
}

func runExercises() {
	fmt.Println("Exercise 1 (Karatsuba 31415926*27182818):", ex1Karatsuba())
	fmt.Println("Exercise 2 (modexp 5^117 mod 19):", ex2ModExp())
	fmt.Println("Exercise 3 (check Karatsuba correctness):", ex3Check())
}
