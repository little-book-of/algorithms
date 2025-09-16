package main

import "testing"

func TestUnsignedWrap(t *testing.T) {
	if got := AddUnsignedBits(255, 1, 8); got != 0 {
		t.Fatalf("u8 255+1 => %d, want 0", got)
	}
	if got := AddUnsignedBits(15, 1, 4); got != 0 {
		t.Fatalf("u4 15+1 => %d, want 0", got)
	}
	if got := AddUnsignedBits(65535, 1, 16); got != 0 {
		t.Fatalf("u16 65535+1 => %d, want 0", got)
	}
}

func TestSignedWrap(t *testing.T) {
	if got := AddSignedBits(127, 1, 8); got != -128 {
		t.Fatalf("s8 127+1 => %d, want -128", got)
	}
	if got := AddSignedBits(-1, -1, 8); got != -2 {
		t.Fatalf("s8 -1+(-1) => %d, want -2", got)
	}
}

func TestToSignedBits(t *testing.T) {
	if got := ToSignedBits(0x80, 8); got != -128 {
		t.Fatalf("toSigned 0x80 => %d, want -128", got)
	}
	if got := ToSignedBits(0x7F, 8); got != 127 {
		t.Fatalf("toSigned 0x7F => %d, want 127", got)
	}
}

func TestFloatBasics(t *testing.T) {
	sum := RepeatAdd(0.1, 10)
	if !AlmostEqual(sum, 1.0, 1e-9) {
		t.Fatal("sum of ten 0.1 should be ~1.0")
	}
}

func TestExercises(t *testing.T) {
	a, b, c := Ex1()
	if a != 15 || b != 0 || c != 1 {
		t.Fatalf("Ex1 got %v %v %v", a, b, c)
	}
	sum, eq1, almost := Ex2()
	if eq1 {
		t.Fatal("Ex2 eq1 should be false for float drift")
	}
	if !almost {
		t.Fatal("Ex2 almost should be true with epsilon")
	}
	x, y := Ex3()
	if x != -128 || y != -2 {
		t.Fatalf("Ex3 got %v %v", x, y)
	}
	if Ex4() != 0 {
		t.Fatal("Ex4 expected 0")
	}
}
