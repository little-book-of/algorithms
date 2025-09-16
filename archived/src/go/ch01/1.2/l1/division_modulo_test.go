package main

import "testing"

func TestDivmodIdentity(t *testing.T) {
	cases := [][2]int{{47, 5}, {23, 7}, {100, 9}}
	for _, c := range cases {
		n, d := c[0], c[1]
		q, r := n/d, n%d
		if n != d*q+r {
			t.Fatalf("identity failed for %d,%d: got %d*%d+%d", n, d, d, q, r)
		}
	}
}

func TestModuloExamples(t *testing.T) {
	if got := hashingExample(1234, 10); got != 4 {
		t.Fatalf("hashingExample: want 4 got %d", got)
	}
	if got := cyclicDay(5, 4); got != 2 {
		t.Fatalf("cyclicDay: want 2 got %d", got)
	}
	if got := modexp(7, 128, 13); got != 3 {
		t.Fatalf("modexp: want 3 got %d", got)
	}
}

func TestEfficiencyBitmask(t *testing.T) {
	cases := []int{5, 12, 20}
	for _, n := range cases {
		if n%8 != n&7 {
			t.Fatalf("bitmask mismatch for %d: %d vs %d", n, n%8, n&7)
		}
	}
}

func TestExercises(t *testing.T) {
	if q, r := ex1(); q != 11 || r != 1 {
		t.Fatalf("ex1: got (%d,%d)", q, r)
	}
	if q, r := ex2(); q != 11 || r != 2 {
		t.Fatalf("ex2: got (%d,%d)", q, r)
	}
	if !ex3(200, 23) {
		t.Fatal("ex3 failed for (200,23)")
	}
	if ex4(37) != 5 {
		t.Fatalf("ex4: want 5 got %d", ex4(37))
	}
	if !ex5(35) {
		t.Fatal("ex5: 35 should be divisible by 7")
	}
}
