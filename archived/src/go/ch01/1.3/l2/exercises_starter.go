package main

import "fmt"

func Ex1() int64 {
	inv, err := InvMod(7, 13) // should be 2
	if err != nil {
		return -1
	}
	return inv
}

func Ex2() (uint64, bool) {
	phi10, _ := PhiU64(10) // 4
	// Verify Euler’s theorem for a=3, n=10: 3^phi(10) ≡ 1 (mod 10)
	n := uint64(10)
	a := uint64(3) % n
	e := phi10
	acc := uint64(1)
	for e > 0 {
		if e&1 == 1 {
			acc = (acc * a) % n
		}
		a = (a * a) % n
		e >>= 1
	}
	return phi10, acc == 1
}

func Ex3() uint64 {
	// Fermat on 341 with base 2: returns 1 though composite
	n := uint64(341)
	e := uint64(340)
	acc := uint64(1)
	base := uint64(2) % n
	for e > 0 {
		if e&1 == 1 {
			acc = (acc * base) % n
		}
		base = (base * base) % n
		e >>= 1
	}
	return acc
}

func Ex4() []uint64 {
	out := make([]uint64, 0)
	for _, n := range []uint64{97, 341, 561, 569} {
		if IsProbablePrimeU64(n) {
			out = append(out, n)
		}
	}
	return out
}

func Ex5() (uint64, uint64) {
	// x≡1 (mod 4), x≡2 (mod 5), x≡3 (mod 7) -> 52 mod 140
	x12, m12, _ := CRTPairU64(1, 4, 2, 5)
	sol, mod, _ := CRTPairU64(x12, m12, 3, 7)
	return sol, mod
}

func runExercises() {
	fmt.Println("Exercise 1 invmod(7,13):", Ex1())
	phi10, ok := Ex2()
	fmt.Println("Exercise 2 phi(10) & check:", phi10, ok)
	fmt.Println("Exercise 3 Fermat 341 result:", Ex3())
	fmt.Println("Exercise 4 probable primes:", Ex4())
	sol, mod := Ex5()
	fmt.Println("Exercise 5 CRT:", sol, "mod", mod)
}