package main

func IsEven(n int) bool { return n%2 == 0 }
func IsOdd(n int) bool  { return n%2 != 0 }

// ParityBit returns "even" or "odd" using the last binary bit.
func ParityBit(n int) string {
	if n&1 == 1 {
		return "odd"
	}
	return "even"
}
