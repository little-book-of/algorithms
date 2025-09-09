import Ch01_1_2_L2.Karatsuba
import Ch01_1_2_L2.ModExp
import Ch01_1_2_L2.Exercises

open Ch01_1_2_L2

def main : IO Unit := do
  -- Baseline multiplication vs. Karatsuba
  let a := (1234 : Nat)
  let b := (5678 : Nat)
  IO.println s!"Naive (builtin)  : {a*b}"
  IO.println s!"Karatsuba result : {karatsuba a b}"

  -- Modular exponent demo
  IO.println s!"modexp(7,128,13) = {modexp 7 128 13}"

  -- Exercises output
  runExercises
