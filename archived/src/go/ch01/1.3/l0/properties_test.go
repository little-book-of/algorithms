package main

import "testing"

func TestParity(t *testing.T) {
	if !IsEven(10) { t.Fatal("10 should be even") }
	if !IsOdd(7)  { t.Fatal("7 should be odd") }
	if ParityBit(11) != "odd" { t.Fatal("11 should be odd by bit") }
	if ParityBit(8)  != "even" { t.Fatal("8 should be even by bit") }
}

func TestDivisibility(t *testing.T) {
	if !IsDivisible(12,3) { t.Fatal("12 % 3 == 0") }
	if  IsDivisible(14,5) { t.Fatal("14 % 5 != 0") }
	if LastDecimalDigit(1234) != 4 { t.Fatal("last digit") }
	if !DivisibleBy10(120) { t.Fatal("120 is divisible by 10") }
}

func TestRemainderAndClock(t *testing.T) {
	q, r := DivIdentity(17, 5)
	if q != 3 || r != 2 { t.Fatalf("div identity got (%d,%d)", q, r) }
	if WeekShift(5, 10) != 1 { t.Fatal("Sat + 10 should be Mon(1)") }
	if PowMod(2, 15, 10) != 8 { t.Fatal("last digit 2^15 should be 8") }
}

func TestExercises(t *testing.T) {
	if Exercise1(42) != "even" { t.Fatal("ex1") }
	if !Exercise2() { t.Fatal("ex2") }
	if Exercise3() != 1 { t.Fatal("ex3") }
	if Exercise4() != 1 { t.Fatal("ex4") }
	if Exercise5() != 8 { t.Fatal("ex5") }
}
