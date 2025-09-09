namespace Ch01_1_3_L2

/-- Extended Euclid on integers:
returns (g, x, y) with a*x + b*y = g = gcd(a,b) up to sign. -/
partial def extendedGCD (a b : Int) : Int × Int × Int := Id.run do
  let mut oldR := a;  let mut r := b
  let mut oldS := (1 : Int); let mut s := (0 : Int)
  let mut oldT := (0 : Int); let mut t := (1 : Int)
  while r ≠ 0 do
    let q := oldR / r
    let nr := oldR - q * r; oldR := r; r := nr
    let ns := oldS - q * s; oldS := s; s := ns
    let nt := oldT - q * t; oldT := t; t := nt
  -- oldR is (±)gcd; Bézout: a*oldS + b*oldT = oldR
  (oldR, oldS, oldT)

/-- Modular inverse for Int `a` modulo positive Nat `m`.
Returns x in [0..m-1] with (a*x) % m = 1, or none if no inverse. -/
def invMod? (a : Int) (m : Nat) : Option Nat :=
  if m = 0 then none else
  let (g, x, _) := extendedGCD a (Int.ofNat m)
  if g.natAbs ≠ 1 then none
  else
    let xm := (x % m).toNat
    some xm

end Ch01_1_3_L2