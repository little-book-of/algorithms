namespace Ch01_1_2_L1

def ex1 : Nat × Nat := (100 / 9, 100 % 9)
def ex2 : Nat × Nat := (123 / 11, 123 % 11)

/-- Check identity n = d*(n/d) + (n%d). -/
def ex3 (n d : Nat) : Bool :=
  let q := n / d
  let r := n % d
  n = d*q + r

/-- Power-of-two modulo via bitmask idea (report arithmetic result). -/
def ex4 (n : Nat) : Nat := n % 16

/-- Simple divisibility test by 7 using %, not a proof. -/
def ex5 (n : Nat) : Bool := n % 7 = 0

def runExercises : IO Unit := do
  let (q1, r1) := ex1
  IO.println s!"Exercise 1 (100//9,100%9): {q1} {r1}"
  let (q2, r2) := ex2
  IO.println s!"Exercise 2 (123//11,123%11): {q2} {r2}"
  IO.println s!"Exercise 3 check (200,23): {ex3 200 23}"
  IO.println s!"Exercise 4 (37 % 16): {ex4 37}"
  IO.println s!"Exercise 5 (35 divisible by 7?): {ex5 35}"

end Ch01_1_2_L1
