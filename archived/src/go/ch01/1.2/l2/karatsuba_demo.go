package main

import "fmt"

func runKaratsubaDemo() {
	a, b := 1234, 5678
	res := Karatsuba(a, b)
	fmt.Println("Karatsuba result:", res)
	fmt.Println("Expected:", a*b)
}
