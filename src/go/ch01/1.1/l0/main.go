package main

import "fmt"

func main() {
	// Demo: decimal -> binary
	n := uint(42)
	fmt.Println("Decimal:", n)
	fmt.Println("Binary :", DecToBin(n))

	// Demo: binary string -> decimal
	s := "1011"
	v, err := BinToDec(s)
	if err != nil {
		fmt.Printf("Invalid binary string: %q\n", s)
	} else {
		fmt.Printf("Binary string %q -> decimal: %d\n", s, v)
	}

	// Run small exercises
	runExercises()
}
