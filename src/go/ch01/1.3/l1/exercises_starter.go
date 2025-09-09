package main

import "fmt"

func Ex1() int64 { return GCD(252, 198) }     // gcd(252,198) = 18
func Ex2() int64 { return LCM(12, 18) }       // 36
func Ex3() (int64, int64) {                   // modular add check
	lhs := (37 + 85) % 12
	rhs := ModAdd(37, 85, 12)
	return lhs, rhs
}
func Ex4() Fraction { return ReduceFraction(84, 126) } // 2/3
func Ex5() int64    { return LCM(12, 18) }             // 36

func runExercises() {
	fmt.Println("Exercise 1 gcd(252,198):", Ex1())
	fmt.Println("Exercise 2 lcm(12,18):", Ex2())
	lhs, rhs := Ex3()
	fmt.Println("Exercise 3 modular add check:", lhs, rhs)
	r := Ex4()
	fmt.Printf("Exercise 4 reduce 84/126: %d/%d\n", r.Num, r.Den)
	fmt.Println("Exercise 5 smallest day multiple:", Ex5())
}