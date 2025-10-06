import Ch01_1_3_L0.Parity
import Ch01_1_3_L0.Divisibility
import Ch01_1_3_L0.Remainder

open Ch01_1_3_L0

namespace Ch01_1_3_L0

def ex1 (n : Nat) : String :=
  if isEven n then "even" else "odd"

def ex2 : Bool :=
  isDivisible 91 7

def ex3 : Nat :=
  (divIdentity 100 9).snd  -- remainder

def ex4 : Nat :=
  weekShift 5 10  -- Saturday(5) + 10 -> Monday(1)

def ex5 : Nat :=
  powMod 2 15 10  -- last digit of 2^15

def runExercises : IO Unit := do
  IO.println s!"Exercise 1 (42 parity): {ex1 42}"
  IO.println s!"Exercise 2 (91 divisible by 7?): {ex2}"
  IO.println s!"Exercise 3 (100 % 9): {ex3}"
  IO.println s!"Exercise 4 (Sat+10): {ex4}"
  IO.println s!"Exercise 5 (last digit of 2^15): {ex5}"

end Ch01_1_3_L0
