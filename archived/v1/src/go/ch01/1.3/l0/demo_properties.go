package main

import "fmt"

func main() {
	fmt.Println("Parity:")
	for _, n := range []int{10, 7} {
		fmt.Printf("  %d is_even? %v  is_odd? %v  (bit) %s\n",
			n, IsEven(n), IsOdd(n), ParityBit(n))
	}

	fmt.Println("\nDivisibility:")
	fmt.Printf("  12 divisible by 3? %v\n", IsDivisible(12, 3))
	fmt.Printf("  14 divisible by 5? %v\n", IsDivisible(14, 5))

	fmt.Println("\nRemainders & identity:")
	q, r := DivIdentity(17, 5)
	fmt.Printf("  17 = 5*%d + %d\n", q, r)

	fmt.Println("\nClock (7-day week):")
	// 0=Mon, 5=Sat; Sat + 10 -> Mon(1)
	fmt.Println("  Saturday(5) + 10 days ->", WeekShift(5, 10))

	fmt.Println("\nLast digit of 2^15:")
	fmt.Println("  ", PowMod(2, 15, 10))
}
