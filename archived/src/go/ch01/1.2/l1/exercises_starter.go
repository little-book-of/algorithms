package main

import "fmt"

func ex1() (int, int) { return 100 / 9, 100 % 9 }
func ex2() (int, int) { return 123 / 11, 123 % 11 }
func ex3(n, d int) bool {
	q, r := n/d, n%d
	return n == d*q+r
}
func ex4(n int) int { return n & 15 }        // n % 16
func ex5(n int) bool { return n%7 == 0 }     // divisibility by 7 (simple)

func runExercises() {
	q, r := ex1()
	fmt.Println("Exercise 1 (100//9,100%9):", q, r)
	q, r = ex2()
	fmt.Println("Exercise 2 (123//11,123%11):", q, r)
	fmt.Println("Exercise 3 check (200,23):", ex3(200, 23))
	fmt.Println("Exercise 4 (37 % 16 via bitmask):", ex4(37))
	fmt.Println("Exercise 5 (35 divisible by 7?):", ex5(35))
}
