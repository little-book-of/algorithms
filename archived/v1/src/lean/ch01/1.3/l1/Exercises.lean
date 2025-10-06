import Ch01_1_3_L1.GcdLcm
import Ch01_1_3_L1.ModularIdentities
import Ch01_1_3_L1.Fractions

open Ch01_1_3_L1

namespace Ch01_1_3_L1

def ex1 : Nat := gcdInt 252 198            -- 18
def ex2 : Nat := lcmInt 12 18               -- 36
def ex3 : Nat × Nat :=                      -- modular add check
  let lhs : Nat := ( (37 : Int) + 85 ) |> (·.toNat? |>.getD 0) -- not ideal; better compute via mod
  let lhsMod : Nat := ( (37 + 85 : Int) |> (fun s => modNorm s 12) )
  let rhs : Nat := modAdd 37 85 12
  (lhsMod, rhs)
def ex4 : Fraction := reduceFraction 84 126 -- 2/3
def ex5 : Nat := lcmInt 12 18               -- 36

/-- Pretty printer for a fraction. -/
def showFrac (f : Fraction) : String :=
  s!"{f.num}/{f.den}"

def runExercises : IO Unit := do
  IO.println s!"Exercise 1 gcd(252,198): {ex1}"
  IO.println s!"Exercise 2 lcm(12,18): {ex2}"
  let (lhs, rhs) := ex3
  IO.println s!"Exercise 3 modular add check: {lhs} {rhs}"
  let r := ex4
  IO.println s!"Exercise 4 reduce 84/126: {showFrac r}"
  IO.println s!"Exercise 5 smallest day multiple: {ex5}"

end Ch01_1_3_L1