package main

import (
	"math/big"
)

// powModBig computes (a^e mod n) with big.Int (handles overflow safely).
func powModBig(a, e, n *big.Int) *big.Int {
	var res big.Int
	res.SetUint64(1)
	var base big.Int
	base.Mod(a, n)
	var exp big.Int
	exp.Set(e)

	for exp.Sign() > 0 {
		if exp.Bit(0) == 1 {
			res.Mul(&res, &base).Mod(&res, n)
		}
		base.Mul(&base, &base).Mod(&base, n)
		exp.Rsh(&exp, 1)
	}
	return &res
}

func decompose(nMinus1 uint64) (d uint64, s uint32) {
	d = nMinus1
	s = 0
	for (d & 1) == 0 {
		d >>= 1
		s++
	}
	return
}

func tryComposite(base uint64, d uint64, n uint64, s uint32) bool {
	N := new(big.Int).SetUint64(n)
	A := new(big.Int).SetUint64(base % n)
	D := new(big.Int).SetUint64(d)

	x := powModBig(A, D, N)
	if x.Cmp(big.NewInt(1)) == 0 || x.Cmp(new(big.Int).Sub(N, big.NewInt(1))) == 0 {
		return false
	}
	for i := uint32(1); i < s; i++ {
		x.Mul(x, x).Mod(x, N)
		if x.Cmp(new(big.Int).Sub(N, big.NewInt(1))) == 0 {
			return false
		}
	}
	return true
}

// IsProbablePrimeU64: Miller–Rabin deterministic for 64-bit with fixed bases.
func IsProbablePrimeU64(n uint64) bool {
	if n < 2 {
		return false
	}
	small := []uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
	for _, p := range small {
		if n == p {
			return true
		}
		if n%p == 0 {
			return false
		}
	}
	d, s := decompose(n - 1)
	bases := []uint64{2, 3, 5, 7, 11, 13, 17}
	for _, a := range bases {
		aa := a % n
		if aa == 0 {
			continue
		}
		if tryComposite(aa, d, n, s) {
			return false
		}
	}
	return true
}