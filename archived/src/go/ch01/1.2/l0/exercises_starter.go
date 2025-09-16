package main

import "fmt"

func exercise1() int { return 326 + 589 }
func exercise2() int { return 704 - 259 }
func exercise3() int { return 38 * 12 }
func exercise4() (q, r int) {
	return 123 / 7, 123 % 7
}

func main() {
	// Run binary demo first (separate from basic ops main output)
	demoBinary()

	fmt.Println("Exercise 1 (326+589):", exercise1())
	fmt.Println("Exercise 2 (704-259):", exercise2())
	fmt.Println("Exercise 3 (38*12):", exercise3())
	q, r := exercise4()
	fmt.Println("Exercise 4 (123//7, 123%7):", q, r)
}
