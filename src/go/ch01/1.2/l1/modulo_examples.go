package main

import "fmt"

// hashingExample maps a key into a table of given size.
func hashingExample(key, size int) int { return key % size }

// cyclicDay wraps weekday (0–6) with a shift.
func cyclicDay(start, shift int) int { return (start + shift) % 7 }

// modexp computes (a^b) mod m using square-and-multiply.
func modexp(a, b, m int) int {
	res := 1
	base := a % m
	exp := b
	for exp > 0 {
		if exp&1 == 1 {
			res = (res * base) % m
		}
		base = (base * base) % m
		exp >>= 1
	}
	return res
}

func runModuloExamples() {
	fmt.Println("Hashing: key=1234, size=10 ->", hashingExample(1234, 10))
	fmt.Println("Cyclic: Saturday(5)+4 ->", cyclicDay(5, 4)) // 2 = Wed
	fmt.Println("modexp(7,128,13) =", modexp(7, 128, 13))   // 3
}
