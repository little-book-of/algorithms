package main

// DecToBin converts a non-negative integer to its binary string (no "0b" prefix).
func DecToBin(n uint) string {
	if n == 0 {
		return "0"
	}
	// collect remainders, then reverse
	var buf [64]byte // enough for 64-bit uint
	i := 0
	for n > 0 {
		if n%2 == 0 {
			buf[i] = '0'
		} else {
			buf[i] = '1'
		}
		i++
		n /= 2
	}
	// reverse in-place portion [0, i)
	for l, r := 0, i-1; l < r; l, r = l+1, r-1 {
		buf[l], buf[r] = buf[r], buf[l]
	}
	return string(buf[:i])
}
