package main

// Unsigned addition with wraparound modulo 2^bits.
func AddUnsignedBits(x, y uint32, bits int) uint32 {
	if bits >= 32 {
		return x + y
	}
	mask := uint32((1 << bits) - 1)
	return (x + y) & mask
}

// Interpret 'value' (0..2^bits-1) as signed two's-complement.
func ToSignedBits(value uint32, bits int) int32 {
	if bits >= 32 {
		return int32(value)
	}
	mask := uint32((1 << bits) - 1)
	v := value & mask
	sign := uint32(1 << (bits - 1))
	if v&sign != 0 {
		// negative: subtract 2^bits
		return int32(v - (sign << 1))
	}
	return int32(v)
}

// Signed two's-complement addition with wraparound at 'bits'.
func AddSignedBits(x, y int32, bits int) int32 {
	if bits >= 32 {
		return x + y
	}
	mask := uint32((1 << bits) - 1)
	raw := (uint32(x) + uint32(y)) & mask
	return ToSignedBits(raw, bits)
}

// 8-bit add returning result and flags (CF for unsigned carry, OF for signed overflow).
func AddWithFlags8(x, y uint8) (result uint8, cf, of bool) {
	wide := uint16(x) + uint16(y)
	r := uint8(wide & 0xFF)
	cf = wide > 0xFF

	sx, sy, sr := int8(x), int8(y), int8(r)
	of = (sx >= 0 && sy >= 0 && sr < 0) || (sx < 0 && sy < 0 && sr >= 0)

	return r, cf, of
}
