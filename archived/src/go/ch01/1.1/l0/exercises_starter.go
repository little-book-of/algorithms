package main

import "fmt"

func runExercises() {
	// Exercise 1: 19 -> binary
	fmt.Println("Exercise 1 (19 -> binary):", DecToBin(19)) // expect 10011

	// Exercise 2: "10101" -> decimal
	if v, err := BinToDec("10101"); err == nil {
		fmt.Println("Exercise 2 ('10101' -> dec):", v) // expect 21
	} else {
		fmt.Println("Exercise 2 error:", err)
	}

	// Exercise 3: 27 -> binary
	fmt.Println("Exercise 3 (27 -> binary):", DecToBin(27)) // expect 11011

	// Exercise 4: verify 0b111111 == 63
	if v, err := BinToDec("111111"); err == nil {
		fmt.Printf("Exercise 4: 111111 -> %d == 63? %v\n", v, v == 63)
	} else {
		fmt.Println("Exercise 4 error:", err)
	}

	// Exercise 5: why binary? (short explanation)
	fmt.Println("Exercise 5: Electronics are reliable with two states (0/1);",
		"binary is simpler and more robust for circuits than base 10.")
}
