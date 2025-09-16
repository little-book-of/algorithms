package main

import "fmt"

func Exercise1(n int) string {
	if IsEven(n) {
		return "even"
	}
	return "odd"
}

func Exercise2() bool { return IsDivisible(91, 7) }

func Exercise3() int {
	_, r := DivIdentity(100, 9)
	return r
}

func Exercise4() int { return WeekShift(5, 10) }         // Sat + 10 -> Mon(1)
func Exercise5() int { return PowMod(2, 15, 10) }        // last digit of 2^15

func runExercises() {
	fmt.Println("Exercise 1 (42 parity):", Exercise1(42))
	fmt.Println("Exercise 2 (91 divisible by 7?):", Exercise2())
	fmt.Println("Exercise 3 (100 % 9):", Exercise3())
	fmt.Println("Exercise 4 (Sat+10):", Exercise4())
	fmt.Println("Exercise 5 (last digit of 2^15):", Exercise5())
}
