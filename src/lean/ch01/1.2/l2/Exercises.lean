import Ch01_1_2_L2.Karatsuba
import Ch01_1_2_L2.ModExp

open Ch01_1_2_L2

namespace Ch01_1_2_L2

def ex1Karatsuba : Nat :=
  karatsuba 31415926 27182818

def ex2ModExp : Nat :=
  modexp 5 117 19

def ex3Check : Bool :=
  let a := (12345 : Nat); let b := (67890 : Nat)
  karatsuba a b == a * b

def runExercises : IO Unit := do
  IO.println s!"Exercise 1 (Karatsuba 31415926*27182818): {ex1Karatsuba}"
  IO.println s!"Exercise 2 (modexp 5^117 mod 19): {ex2ModExp}"
  IO.println s!"Exercise 3 (check Karatsuba correctness): {ex3Check}"

end Ch01_1_2_L2
