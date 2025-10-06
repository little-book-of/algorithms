package main

import "fmt"

func runExercises() {
	// Exercise 1: 100 in octal/hex
	n := uint(100)
	fmt.Printf("Exercise 1 (100): oct=%o hex=%X\n", n, n) // expect oct=144 hex=64

	// Exercise 2: -7 in 8-bit two's complement
	tc := ToTwosComplement(-7, 8)
	fmt.Println("Exercise 2 (-7 in 8-bit):", tc) // expect 11111001

	// Exercise 3: verify 0xFF == 255
	ok := (0xFF == 255)
	fmt.Println("Exercise 3 (0xFF == 255?):", ok) // expect true

	// Exercise 4: parse "11111001" as 8-bit two's complement
	v, err := FromTwosComplement("11111001")
	if err != nil {
		fmt.Println("Exercise 4 error:", err)
	} else {
		fmt.Println("Exercise 4 (parse '11111001'):", v) // expect -7
	}
}
