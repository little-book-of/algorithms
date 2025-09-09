package main

import (
	"fmt"
	"math/big"
)

func decimalExactDemo() {
	// Exact decimal with big.Rat
	a := new(big.Rat).SetString("0.1")
	b := new(big.Rat).SetString("0.2")
	c := new(big.Rat).SetString("0.3")
	if a == nil || b == nil || c == nil {
		panic("big.Rat SetString failed")
	}
	sum := new(big.Rat).Add(a, b)
	fmt.Println("big.Rat 0.1 + 0.2 =", sum.RatString())
	fmt.Println("Equal to 0.3?", sum.Cmp(c) == 0)

	// Higher precision binary floating point with big.Float (not exact decimal, but controllable precision)
	fa, fb := new(big.Float).SetPrec(200).SetFloat64(0.1), new(big.Float).SetPrec(200).SetFloat64(0.2)
	fs := new(big.Float).SetPrec(200).Add(fa, fb)
	fmt.Println("big.Float(0.1)+big.Float(0.2) =", fs.Text('g', -1))
}
