package main

import "testing"

func TestToTwosComplement(t *testing.T) {
	cases := []struct {
		x    int64
		bits int
		want string
	}{
		{5, 8, "00000101"},
		{-5, 8, "11111011"},
		{-7, 8, "11111001"},
		{0, 1, "0"},
		{-1, 8, "11111111"},
	}
	for _, c := range cases {
		got := ToTwosComplement(c.x, c.bits)
		if got != c.want {
			t.Fatalf("ToTwosComplement(%d,%d): want %q, got %q", c.x, c.bits, c.want, got)
		}
	}
}

func TestFromTwosComplement(t *testing.T) {
	cases := []struct {
		in      string
		want    int64
		wantErr bool
	}{
		{"00000101", 5, false},
		{"11111011", -5, false},
		{"11111001", -7, false},
		{"", 0, true},
		{"2", 0, true},
		{"abc", 0, true},
	}
	for _, c := range cases {
		got, err := FromTwosComplement(c.in)
		if c.wantErr {
			if err == nil {
				t.Fatalf("FromTwosComplement(%q): expected error, got %d", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("FromTwosComplement(%q): unexpected error: %v", c.in, err)
		}
		if got != c.want {
			t.Fatalf("FromTwosComplement(%q): want %d, got %d", c.in, c.want, got)
		}
	}
}
