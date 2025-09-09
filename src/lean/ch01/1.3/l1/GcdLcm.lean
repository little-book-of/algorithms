namespace Ch01_1_3_L1

/-- Euclid's algorithm on natural numbers. -/
partial def gcdNat (a b : Nat) : Nat :=
  if b = 0 then a else gcdNat b (a % b)

/-- Least common multiple on natural numbers; lcm(0, b) = 0. -/
def lcmNat (a b : Nat) : Nat :=
  if a = 0 || b = 0 then 0
  else
    let g := gcdNat a b
    (a / g) * b

/-- Convenience: gcd on integers returns a natural (nonnegative). -/
def gcdInt (a b : Int) : Nat :=
  gcdNat a.natAbs b.natAbs

/-- Convenience: lcm on integers (uses absolute values). -/
def lcmInt (a b : Int) : Nat :=
  lcmNat a.natAbs b.natAbs

end Ch01_1_3_L1