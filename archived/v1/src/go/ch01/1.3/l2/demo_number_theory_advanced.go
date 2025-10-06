package main

import "fmt"

func main() {
	// Extended Euclid / inverse
	g, x, y := ExtendedGCD(240, 46)
	fmt.Println("extended_gcd(240,46) ->", g, x, y)
	if inv, err := InvMod(3, 7); err == nil {
		fmt.Println("invmod(3,7) =", inv)
	} else {
		fmt.Println("invmod(3,7) failed:", err)
	}

	// Totient
	if v, _ := PhiU64(10); true {
		fmt.Println("phi(10) =", v)
	}
	if v, _ := PhiU64(36); true {
		fmt.Println("phi(36) =", v)
	}

	// Miller–Rabin
	for _, n := range []uint64{97, 561, 1105, 2147483647} {
		fmt.Printf("is_probable_prime(%d) = %v\n", n, IsProbablePrimeU64(n))
	}

	// CRT: x≡2 (mod 3), x≡3 (mod 5), x≡2 (mod 7) -> 23 mod 105
	x12, m12, _ := CRTPairU64(2, 3, 3, 5)
	res, mod, _ := CRTPairU64(x12, m12, 2, 7)
	fmt.Println("CRT ->", res, "mod", mod)

	// Exercises (optional to print)
	runExercises()
}