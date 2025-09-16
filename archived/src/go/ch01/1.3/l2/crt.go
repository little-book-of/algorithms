package main

import (
	"errors"
)

// CRTPairU64 solves x ≡ a1 (mod m1), x ≡ a2 (mod m2) for coprime moduli.
// Returns (x, M=m1*m2) with x in [0, M-1].
func CRTPairU64(a1, m1, a2, m2 uint64) (uint64, uint64, error) {
	// Require coprime moduli: check via gcd on int64 domain (safe for <= 2^63-1).
	// If you need full u64 gcd, add a small uint64 gcd; for this teaching code,
	// we rely on typical small moduli.
	inv, err := InvMod(int64(m1%m2), int64(m2))
	if err != nil {
		return 0, 0, errors.New("moduli must be coprime for CRT (inv failed)")
	}
	k := ((a2 + m2 - (a1 % m2)) % m2)
	kk := (uint64(inv) % m2)
	k = (k * kk) % m2

	M := m1 * m2
	x := (a1 + k*m1) % M
	return x, M, nil
}

// CRT folds CRTPairU64 across a list of congruences with pairwise coprime moduli.
func CRT(congruences [][2]uint64) (uint64, uint64, error) {
	if len(congruences) == 0 {
		return 0, 0, errors.New("no congruences")
	}
	a, m := congruences[0][0], congruences[0][1]
	for i := 1; i < len(congruences); i++ {
		var err error
		a, m, err = CRTPairU64(a, m, congruences[i][0], congruences[i][1])
		if err != nil {
			return 0, 0, err
		}
	}
	return a, m, nil
}