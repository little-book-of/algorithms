package main

import "testing"

func TestKaratsubaSmall(t *testing.T) {
	cases := [][2]int{{3, 7}, {12, 34}, {123, 456}, {1234, 5678}}
	for _, c := range cases {
		a, b := c[0], c[1]
		got := Karatsuba(a, b)
		want := a * b
		if got != want {
			t.Fatalf("Karatsuba(%d,%d) got %d want %d", a, b, got, want)
		}
	}
}

func TestBitLen(t *testing.T) {
	if bitLen(0) != 0 || bitLen(1) != 1 || bitLen(2) != 2 || bitLen(3) != 2 {
		t.Fatal("bitLen wrong")
	}
}

func TestModExp(t *testing.T) {
	if got := ModExp(7, 128, 13); got != 3 {
		t.Fatalf("ModExp(7,128,13) got %d want 3", got)
	}
	if got := ModExp(5, 117, 19); got != 1 { // known result
		t.Fatalf("ModExp(5,117,19) got %d want 1", got)
	}
}
