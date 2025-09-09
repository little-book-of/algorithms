package main

import "errors"

// BinToDec parses a binary string like "1011" into a non-negative integer.
func BinToDec(s string) (uint, error) {
	if len(s) == 0 {
		return 0, errors.New("empty string")
	}
	var v uint
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '0':
			v = v << 1
		case '1':
			v = (v << 1) | 1
		default:
			return 0, errors.New("invalid character (expected '0' or '1')")
		}
	}
	return v, nil
}
