package main

import "fmt"

// For power-of-two modulus m = 2^k, n % m == n & (m-1)
func runEfficiencyDemo() {
	for _, n := range []int{5, 12, 20} {
		fmt.Printf("%d %% 8 = %d, bitmask = %d\n", n, n%8, n&7)
	}
}
