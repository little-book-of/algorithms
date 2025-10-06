package main

import "testing"

func TestBasicOps(t *testing.T) {
	if 478+259 != 737 {
		t.Fatalf("addition wrong")
	}
	if 503-78 != 425 {
		t.Fatalf("subtraction wrong")
	}
	if 214*3 != 642 {
		t.Fatalf("multiplication wrong")
	}
	if q, r := 47/5, 47%5; q != 9 || r != 2 {
		t.Fatalf("division wrong: q=%d r=%d", q, r)
	}
}

func TestExercises(t *testing.T) {
	if exercise1() != 915 {
		t.Fatalf("exercise1")
	}
	if exercise2() != 445 {
		t.Fatalf("exercise2")
	}
	if exercise3() != 456 {
		t.Fatalf("exercise3")
	}
	q, r := exercise4()
	if q != 17 || r != 4 {
		t.Fatalf("exercise4: q=%d r=%d", q, r)
	}
}
