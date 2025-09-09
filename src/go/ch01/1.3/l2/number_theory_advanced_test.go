package main

import "testing"

func TestExtendedGCDInv(t *testing.T) {
	g, x, y := ExtendedGCD(240, 46)
	if int(g) != 2 || 240*x+46*y != g {
		t.Fatal("extended gcd wrong")
	}
	if inv, err := InvMod(3, 7); err != nil || inv != 5 {
		t.Fatal("InvMod(3,7) should be 5")
	}
	if _, err := InvMod(6, 9); err == nil {
		t.Fatal("InvMod should fail for non-coprime")
	}
}

func TestPhi(t *testing.T) {
	if v, _ := PhiU64(1); v != 1 {
		t.Fatal("phi(1)")
	}
	if v, _ := PhiU64(10); v != 4 {
		t.Fatal("phi(10)")
	}
	if v, _ := PhiU64(36); v != 12 {
		t.Fatal("phi(36)")
	}
	if v, _ := PhiU64(97); v != 96 {
		t.Fatal("phi(97)")
	}
}

func TestMillerRabin(t *testing.T) {
	for _, p := range []uint64{2, 3, 5, 7, 11, 97, 569, 2147483647} {
		if !IsProbablePrimeU64(p) {
			t.Fatalf("%d should be prime", p)
		}
	}
	for _, c := range []uint64{1, 4, 6, 8, 9, 10, 12, 100} {
		if IsProbablePrimeU64(c) {
			t.Fatalf("%d should be composite", c)
		}
	}
	for _, carm := range []uint64{561, 1105, 1729, 2465} {
		if IsProbablePrimeU64(carm) {
			t.Fatalf("%d is Carmichael; MR should reject", carm)
		}
	}
}

func TestCRT(t *testing.T) {
	x, m, err := CRT([][2]uint64{{2, 3}, {3, 5}, {2, 7}})
	if err != nil {
		t.Fatal(err)
	}
	if x%3 != 2 || x%5 != 3 || x%7 != 2 || m != 105 {
		t.Fatal("CRT mismatch")
	}
}

func TestExercises(t *testing.T) {
	if Ex1() != 2 {
		t.Fatal("Ex1")
	}
	phi10, ok := Ex2()
	if phi10 != 4 || !ok {
		t.Fatal("Ex2")
	}
	if Ex3() != 1 {
		t.Fatal("Ex3")
	}
	got := Ex4()
	want := []uint64{97, 569}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("Ex4 got %v want %v", got, want)
	}
	sol, mod := Ex5()
	if sol != 52 || mod != 140 {
		t.Fatalf("Ex5 got %d mod %d want 52 mod 140", sol, mod)
	}
}