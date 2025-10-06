package main

import (
	"fmt"
)

// Ex1: 4-bit counter sequence starting at 14, add 1 three times.
func Ex1() (uint32, uint32, uint32) {
	a := AddUnsignedBits(14, 1, 4) // 15
	b := AddUnsignedBits(a, 1, 4)  // 0
	c := AddUnsignedBits(b, 1, 4)  // 1
	return a, b, c
}

// Ex2: Does 0.1 added ten times equal exactly 1.0?
func Ex2() (sum float64, eq1, almost bool) {
	sum = RepeatAdd(0.1, 10)
	return sum, sum == 1.0, AlmostEqual(sum, 1.0, 1e-9)
}

// Ex3: 8-bit signed addition demos: 127+1 and -1+(-1)
func Ex3() (int32, int32) {
	a := AddSignedBits(127, 1, 8)  // -128
	b := AddSignedBits(-1, -1, 8)  // -2
	return a, b
}

// Ex4: Unsigned 16-bit wrap for 65535 + 1 -> 0
func Ex4() uint32 {
	return AddUnsignedBits(65535, 1, 16)
}

func runExercises() {
	a, b, c := Ex1()
	fmt.Println("Exercise 1 (4-bit sequence from 14):", a, b, c)
	sum, eq1, almost := Ex2()
	fmt.Printf("Exercise 2 (ten*0.1): sum=%.17g eq1=%v almost=%v\n", sum, eq1, almost)
	x, y := Ex3()
	fmt.Println("Exercise 3 (signed adds):", x, y)
	fmt.Println("Exercise 4 (u16 65535+1):", Ex4())
}
