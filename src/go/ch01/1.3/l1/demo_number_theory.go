package main

import "fmt"

func main() {
	// GCD / LCM
	a, b := int64(252), int64(198)
	fmt.Printf("gcd(%d, %d) = %d\n", a, b, GCD(a, b))
	fmt.Printf("lcm(12, 18) = %d\n", LCM(12, 18))

	// Modular identities
	x, y, m := int64(123), int64(456), int64(7)
	fmt.Printf("(x+y)%%m vs ModAdd: %d %d\n", (x+y)%m, ModAdd(x, y, m))
	fmt.Printf("(x*y)%%m vs ModMul: %d %d\n", (x*y)%m, ModMul(x, y, m))

	// Fraction reduction
	r := ReduceFraction(84, 126)
	fmt.Printf("reduceFraction(84,126) = %d/%d\n", r.Num, r.Den)

	// Exercises (optional to show here)
	runExercises()
}