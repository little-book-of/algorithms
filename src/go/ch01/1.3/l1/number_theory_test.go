package main

import "testing"

func TestGCDLCM(t *testing.T) {
	if GCD(20, 14) != 6-4 { // 2
		t.Fatal("gcd(20,14) should be 2")
	}
	if GCD(252, 198) != 18 {
		t.Fatal("gcd(252,198) should be 18")
	}
	if LCM(12, 18) != 36 {
		t.Fatal("lcm(12,18) should be 36")
	}
	if LCM(0, 5) != 0 {
		t.Fatal("lcm(0,5) should be 0")
	}
}

func TestModularIdentities(t *testing.T) {
	a, b, m := int64(123), int64(456), int64(7)
	if ModAdd(a, b, m) != (a+b)%m {
		t.Fatal("ModAdd mismatch")
	}
	if ModSub(a, b, m) != (a-b)%m {
		t.Fatal("ModSub mismatch")
	}
	if ModMul(a, b, m) != (a*b)%m {
		t.Fatal("ModMul mismatch")
	}
	if ModAdd(37, 85, 12) != ((37%12)+(85%12))%12 {
		t.Fatal("ModAdd example mismatch")
	}
}

func TestFractionReduce(t *testing.T) {
	if r := ReduceFraction(84, 126); r.Num != 2 || r.Den != 3 {
		t.Fatalf("ReduceFraction(84,126) got %d/%d", r.Num, r.Den)
	}
	if r := ReduceFraction(-6, 8); r.Num != -3 || r.Den != 4 {
		t.Fatalf("ReduceFraction(-6,8) got %d/%d", r.Num, r.Den)
	}
	if r := ReduceFraction(6, -8); r.Num != -3 || r.Den != 4 {
		t.Fatalf("ReduceFraction(6,-8) got %d/%d", r.Num, r.Den)
	}
	if r := ReduceFraction(-6, -8); r.Num != 3 || r.Den != 4 {
		t.Fatalf("ReduceFraction(-6,-8) got %d/%d", r.Num, r.Den)
	}
}

func TestExercises(t *testing.T) {
	if Ex1() != 18 {
		t.Fatal("Ex1")
	}
	if Ex2() != 36 {
		t.Fatal("Ex2")
	}
	lhs, rhs := Ex3()
	if lhs != rhs {
		t.Fatal("Ex3")
	}
	if r := Ex4(); !(r.Num == 2 && r.Den == 3) {
		t.Fatal("Ex4")
	}
	if Ex5() != 36 {
		t.Fatal("Ex5")
	}
}