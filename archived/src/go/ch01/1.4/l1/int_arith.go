package main

import "math/big"

// Exported bounds for consistency with demos/tests.
const (
	Int32Min int32 = -1 << 31
	Int32Max int32 = 1<<31 - 1
	Int64Min int64 = -1 << 63
	Int64Max int64 = 1<<63 - 1
)

/********** int32 **********/

func AddI32Checked(a, b int32) (int32, bool) {
	s := int64(a) + int64(b)
	if s < int64(Int32Min) || s > int64(Int32Max) {
		return 0, false
	}
	return int32(s), true
}

func AddI32Wrapping(a, b int32) int32 {
	return int32(uint32(a) + uint32(b)) // two's-complement wrap
}

func AddI32Saturating(a, b int32) int32 {
	s := int64(a) + int64(b)
	if s > int64(Int32Max) {
		return Int32Max
	}
	if s < int64(Int32Min) {
		return Int32Min
	}
	return int32(s)
}

func MulI32Checked(a, b int32) (int32, bool) {
	p := int64(a) * int64(b)
	if p < int64(Int32Min) || p > int64(Int32Max) {
		return 0, false
	}
	return int32(p), true
}

/********** int64 **********/

func AddI64Checked(a, b int64) (int64, bool) {
	if (b > 0 && a > Int64Max-b) || (b < 0 && a < Int64Min-b) {
		return 0, false
	}
	return a + b, true
}

func AddI64Wrapping(a, b int64) int64 {
	return int64(uint64(a) + uint64(b)) // explicit wrap
}

func AddI64Saturating(a, b int64) int64 {
	if b > 0 && a > Int64Max-b {
		return Int64Max
	}
	if b < 0 && a < Int64Min-b {
		return Int64Min
	}
	return a + b
}

func MulI64Checked(a, b int64) (int64, bool) {
	z := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	if z.Cmp(big.NewInt(Int64Min)) < 0 || z.Cmp(big.NewInt(Int64Max)) > 0 {
		return 0, false
	}
	return z.Int64(), true
}
