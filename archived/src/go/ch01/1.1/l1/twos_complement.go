package main

import (
	"errors"
	"fmt"
)

// ToTwosComplement encodes x into a two's complement bitstring of given width.
// bits must be in [1, 64]. The value is truncated to the requested width.
func ToTwosComplement(x int64, bits int) string {
	if bits < 1 || bits > 64 {
		panic("bits must be between 1 and 64")
	}
	var mask uint64
	if bits == 64 {
		mask = ^uint64(0)
	} else {
		mask = (1 << bits) - 1
	}
	u := uint64(x) & mask

	// Build string high-to-low.
	out := make([]byte, bits)
	for i := bits - 1; i >= 0; i-- {
		if (u & 1) == 1 {
			out[i] = '1'
		} else {
			out[i] = '0'
		}
		u >>= 1
	}
	return string(out)
}

// FromTwosComplement decodes a two's-complement bitstring (e.g., "11111001")
// into a signed integer. The width is len(s). Returns error on invalid input.
func FromTwosComplement(s string) (int64, error) {
	if len(s) == 0 || len(s) > 64 {
		return 0, errors.New("invalid width")
	}
	var val uint64
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '0':
			val = val << 1
		case '1':
			val = (val << 1) | 1
		default:
			return 0, fmt.Errorf("invalid character %q", s[i])
		}
	}
	bits := uint(len(s))
	top := uint64(1) << (bits - 1)
	if (val & top) == 0 {
		// positive
		return int64(val), nil
	}
	// negative: subtract 2^bits
	full := uint64(1) << bits
	return int64(val - full), nil
}
