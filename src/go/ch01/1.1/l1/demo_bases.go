package main

import (
	"fmt"
	"strings"
)

func main() {
	n := uint(42)

	// Print same value in decimal, octal, hex
	fmt.Println("Decimal:", n)
	fmt.Printf("Octal  : %o\n", n)
	fmt.Printf("Hex    : %X\n", n)

	// “Literals”: Go has 0, 0o (octal), 0x (hex); 0b for binary too.
	octLit := uint(0o52) // 42
	hexLit := uint(0x2A) // 42
	fmt.Println("Octal literal 0o52 ->", octLit)
	fmt.Println("Hex literal 0x2A ->", hexLit)

	// Small demo: two's complement for -7 in 8 bits
	tc := ToTwosComplement(-7, 8)
	fmt.Println("Two's complement (-7, 8-bit):", tc)

	v, err := FromTwosComplement(tc) // infers width from string length
	if err != nil {
		fmt.Println("Decode error:", err)
	} else {
		fmt.Println("Decoded back:", v)
	}

	runExercises()
}

// upper is a tiny helper used in exercises output.
func upper(s string) string { return strings.ToUpper(s) }
