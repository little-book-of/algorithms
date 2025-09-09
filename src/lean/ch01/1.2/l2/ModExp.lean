namespace Ch01_1_2_L2

/-- Modular exponentiation via square-and-multiply (binary exponentiation).
Computes `(a^b) % m` for naturals. -/
def modexp (a b m : Nat) : Nat :=
  if m = 1 then 0 else
  let rec go (base : Nat) (e : Nat) (acc : Nat) : Nat :=
    if e = 0 then acc
    else
      let acc' := if e % 2 = 1 then (acc * base) % m else acc
      go ((base * base) % m) (e / 2) acc'
  go (a % m) b 1

end Ch01_1_2_L2
