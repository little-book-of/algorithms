package main

import "testing"

func TestDecToBin(t *testing.T) {
	cases := []struct {
		in   uint
		want string
	}{
		{0, "0"},
		{1, "1"},
		{2, "10"},
		{5, "101"},
		{19, "10011"},
		{42, "101010"},
	}
	for _, c := range cases {
		if got := DecToBin(c.in); got != c.want {
			t.Fatalf("DecToBin(%d): want %q, got %q", c.in, c.want, got)
		}
	}
}

func TestBinToDec(t *testing.T) {
	cases := []struct {
		in      string
		want    uint
		wantErr bool
	}{
		{"0", 0, false},
		{"1", 1, false},
		{"10", 2, false},
		{"101", 5, false},
		{"10011", 19, false},
		{"101010", 42, false},
		{"", 0, true},
		{"102", 0, true},
		{"abc", 0, true},
	}
	for _, c := range cases {
		got, err := BinToDec(c.in)
		if c.wantErr {
			if err == nil {
				t.Fatalf("BinToDec(%q): expected error, got value %d", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("BinToDec(%q): unexpected error: %v", c.in, err)
		}
		if got != c.want {
			t.Fatalf("BinToDec(%q): want %d, got %d", c.in, c.want, got)
		}
	}
}
