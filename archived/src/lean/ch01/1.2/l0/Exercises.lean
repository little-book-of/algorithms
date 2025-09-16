namespace Ch01_1_2_L0

/-- Exercises mirroring the L0 section. -/
def exercise1 : Nat := 326 + 589
def exercise2 : Nat := 704 - 259
def exercise3 : Nat := 38 * 12
def exercise4 : Nat × Nat := (123 / 7, 123 % 7)

def runExercises : IO Unit := do
  IO.println s!"Exercise 1 (326+589): {exercise1}"
  IO.println s!"Exercise 2 (704-259): {exercise2}"
  IO.println s!"Exercise 3 (38*12): {exercise3}"
  let (q, r) := exercise4
  IO.println s!"Exercise 4 (123//7, 123%7): {q} {r}"

end Ch01_1_2_L0
