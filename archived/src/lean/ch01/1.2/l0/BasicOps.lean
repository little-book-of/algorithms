namespace Ch01_1_2_L0

/-- Addition, subtraction (Nat), multiplication. -/
def add (a b : Nat) : Nat := a + b
def sub (a b : Nat) : Nat := a - b   -- `Nat` subtraction saturates at 0 if b > a
def mul (a b : Nat) : Nat := a * b

/-- Integer division and remainder for natural numbers. -/
def divMod (n d : Nat) : Nat × Nat :=
  (n / d, n % d)

/-- Print the core examples from the text. -/
def demoBasicOps : IO Unit := do
  let a1 := add 478 259
  IO.println s!"Addition: 478 + 259 = {a1}"

  let s1 := sub 503 78
  IO.println s!"Subtraction: 503 - 78 = {s1}"

  let m1 := mul 214 3
  IO.println s!"Multiplication: 214 * 3 = {m1}"

  let (q, r) := divMod 47 5
  IO.println s!"Division: 47 / 5 -> quotient {q}, remainder {r}"

end Ch01_1_2_L0
